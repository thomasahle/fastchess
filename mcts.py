import sys
import chess
import numpy as np
import math
from math import sqrt
import random
import fastchess
import pst
from fastchess import mirror_move
from collections import namedtuple


Args = namedtuple('MctsArgs', ['model', 'debug', 'cpuct', 'legal_t', 'cap_t', 'chk_t'])


class Node:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, parent_board, parent_vec, move, prior, args):
        """ Make a new game node representing the state of pushing `move` to `parent_board`.
            If `move` is None, the node is assumed to be a root node at `parent_board`. """
        self.children = []
        self.args = args
        self.move = move
        self.parent_board = parent_board
        self.parent_vec = parent_vec
        # We expand this as the node is visited
        self.board = None
        self.vec = None
        # Statistics. P is prior, Q is avg reward and N is number of visits.
        self.P = prior
        self.Q = 1
        self.N = 0
        self.game_over = None
        # If we are at the root, make a fake first rollout,
        # just because the parent_board is actually the reeal board, and it's
        # confusing otherwise.
        if move is None:
            # Could do this using null-move?...
            self.board = parent_board
            self.vec = parent_vec
            # The Q value at the root doesn't really matter...
            self.Q, self.game_over = self.eval(self.vec, self.board)
            self.N = 1
            # Even if we think it's game-over (like a repetition), we continue to
            # play of people ask us to.
            self.game_over = False

    def eval(self, vec, board):
        v = {'1-0': 1, '0-1': -1, '1/2-1/2': 0, '*': None}[board.result()]
        if v is not None:
            return (v if board.turn == chess.WHITE else -v), True
        # Result doesn't check for repetitions unless we add claim_draw=True,
        # but even then it doesn't quite do what we want.
        if board.is_repetition(count=2):
            return 0, True
        return self.args.model.get_eval(vec, board), False

    def rollout(self):
        """ Returns the leaf value relative to the current player of the node. """

        self.N += 1

        # Game over won't be set before the board is evaluated.
        if self.game_over:
            return self.Q

        # If first visit, expand board
        if self.N == 1:
            self.vec = self.args.model.apply(
                self.parent_vec.copy(), self.parent_board, self.move)
            # Don't copy the entire move stack, it just takes up memory.
            # We do need some though, to prevent repetition draws.
            # Half-move clock is copied separately
            self.board = self.parent_board.copy(stack=3)
            self.board.push(self.move)

            self.Q, self.game_over = self.eval(self.vec, self.board)
            return self.Q

        # If second visit, expand children
        if self.N == 2:
            for p, move in self.args.model.get_clean_moves(
                    self.board,
                    self.vec,
                    debug=self.args.debug,
                    legal_t=self.args.legal_t,
                    cap_t=self.args.cap_t,
                    chk_t=self.args.chk_t,
                    ):
                self.children.append(Node(self.board, self.vec, move, p, self.args))

        # Find best child (small optimization, since this is actually a bottle neck)
        # TODO: Even now, this is still pretty slow
        # _, node = max((-n.Q + CPUCT * n.P * sqrtN / (1 + n.N), n) for n in self.children)
        # is better, but it fails when two nodes have the same score...
        sqrtN = self.args.cpuct * math.sqrt(self.N)
        node = max(self.children,
                   key=lambda n: -n.Q + n.P * sqrtN / (1 + n.N))

        # Visit it and flip the sign
        s = -node.rollout()
        # Update our own average reward
        self.Q = ((self.N - 1) * self.Q + s) / self.N

        # TODO: Could update prior to safe division?
        # self.P = (self.N - 1) * self.P / self.N

        # Propagate the value further up the tree
        return s
