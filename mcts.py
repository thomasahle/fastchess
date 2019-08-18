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


Args = namedtuple('MctsArgs', ['model', 'debug', 'cpuct'])


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
            self.Q, self.game_over = self.args.model.eval(self.vec, self.board)
            self.N = 1

    def clean_moves(self):
        board = self.board
        moves = []
        scores = []
        vec = self.vec[1 - int(board.turn)]

        if self.args.debug:
            vec1 = self.args.model.from_scratch(board, self.args.debug)[
                1 - int(board.turn)]
            if not np.allclose(vec, vec1, atol=1e-5, rtol=1e-2):
                print(board)
                print(vec1)
                print(vec)
                assert False

        # TODO: Another approach is to use top_k to get the moves
        #       and simply trust that they are legal.
        # self.model.top_k(self.vec)
        # for m in board.generate_legal_moves():
        for m in board.legal_moves:
            moves.append(m)
            cap = board.is_capture(m)
            # TODO: There might be a faster way, inspired by the is_into_check method.
            # or _attackers_mask. Some sort of pseudo-is-check should be sufficient.
            board.push(m)
            chk = board.is_check()
            board.pop()
            # Hack: We make sure that checks and captures are always included,
            # and that no move has a completely non-existent prior.
            prior = vec[1 +
                        self.args.model.move_to_id[m if board.turn else mirror_move(m)]]
            # Add some bonus for being a legal move and check or cap.
            # Maybe these values should be configurable.
            prior += 1 + int(chk or cap)
            scores.append(prior)

        # First entry keeps the word count
        n = vec[0]
        scores = np.array(scores) / n
        scores = np.exp(scores - np.max(scores))
        scores /= np.sum(scores)
        return zip(scores, moves)

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
            self.board = self.parent_board.copy(stack=8)
            self.board.push(self.move)

            self.Q, self.game_over = self.args.model.eval(self.vec, self.board)
            return self.Q

        # If second visit, expand children
        if self.N == 2:
            for p, move in self.clean_moves():
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
