import chess
import numpy as np
import math
from math import sqrt
import random
import fastchess
import pst


CPUCT = 2


class Model:
    def __init__(self, fasttext_model):
        self.fc = fasttext_model

    def eval(self, board, debug=False):
        """ Returns a single score relative to board.turn """
        # If game over, just stop
        if board.is_game_over():
            v = {'1-0':1, '0-1':-1, '1/2-1/2':0}[board.result()]
            return v if board.turn == chess.WHITE else -v

        # We first calculate the value relative to white
        res = 0
        for square, piece in board.piece_map().items():
            p = piece.symbol()
            if p.isupper():
                res += pst.piece[p] + pst.pst[p][63-square]
            else:
                p = p.upper()
                res -= pst.piece[p] + pst.pst[p][square]
        if debug:
            print('Pre norm score:', res)
        # Normalize in [-1, 1]
        res = math.atan(res/100)/math.pi*2
        if debug:
            print('Post norm score:', res)
        # Then flip it to the current player
        return res if board.turn == chess.WHITE else -res

    def predict(self, board, n=40, debug=False):
        """ Returns list of `n` (prob, move) legal pairs """
        pre = {m:p for p,m in self.fc.find_moves(board, n, debug=debug, flipped=False)}
        res = []
        for m in board.generate_legal_moves():
            cap = board.is_capture(m)
            board.push(m)
            chk = board.is_check()
            board.pop()
            # Hack: We make sure that checks and captures are always included,
            # and that no move has a completely non-existent prior.
            p = max(pre.get(m,0), .01, .1*int(chk or cap))
            res.append((p,m))
        return res


class Node:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, parent_board, move, prior, model, debug=False):
        """ Make a new game node representing the state of pushing `move` to `parent_board`.
            If `move` is None, the node is assumed to be a root node at `parent_board`. """
        self.children = []
        self.parent_board = parent_board
        self.move = move
        self.board = None # We expand this as the node is visited
        self.P = prior
        self.Q = 1 # Q, avg reward of node
        self.N = 0 # N, total visit count for node
        self.model = model
        self.debug = debug
        # If we are at the root, make a fake first rollout.
        if move is None:
            self.board = parent_board
            # The Q value at the root doesn't really matter...
            self.Q = model.eval(self.board, debug=debug)
            self.N = 1

    def search(self, rolls):
        """ Do `rolls` rollouts and return the best move. """
        for i in range(rolls):
            self.rollout()
        return max(self.children, key = lambda n: n.N).move

    def rollout(self):
        """ Returns the leaf value relative to the current player of the node. """

        self.N += 1

        if self.board and self.board.is_game_over():
            # If board is set, Q should already be the evaluation, which includes
            # checkmate/stalemate.
            return self.Q

        # If first visit, expand board
        if self.N == 1:
            # Don't copy the move stack, it just takes up memory.
            self.board = self.parent_board.copy(stack = False)
            self.board.push(self.move)
            self.Q = self.model.eval(self.board, debug=self.debug)
            return self.Q

        # If second visit, expand children
        if self.N == 2:
            for p, move in self.model.predict(self.board, debug=self.debug):
                if self.board.is_legal(move):
                    self.children.append(Node(self.board, move, p, self.model))

        # Find best child
        node = max(self.children,
                   key = lambda n: -n.Q + CPUCT * n.P * sqrt(self.N) / (1 + n.N))
        # Visit it
        s = -node.rollout()
        # Note that Q is an average in the newer paper.
        self.Q = ((self.N-1)*self.Q + s)/self.N
        # Return the terminal value, not our own
        return s


