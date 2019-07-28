import chess
import numpy as np
import math
from math import sqrt
import random
import fastchess
import pst

class Model:
    def __init__(self, fasttext_model):
        self.fc = fasttext_model

    def eval(self, board, debug=False):
        """ Returns a single score relative to board.turn """
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
        #print(board)
        #v = res if board.turn == chess.WHITE else -res
        #print('res', v)
        # Then flip it to the current player
        return res if board.turn == chess.WHITE else -res

    def predict(self, board, n=20, debug=False):
        """ Returns list of `n` (prob, move) legal pairs """
        return self.fc.find_moves(board, n, debug=debug, flipped=False)

cpuct = 2

class Node:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, parent_board, move, prior, model, debug=True):
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
            # TODO: The Q value at the root kinda doesn't matter...
            self.Q = model.eval(self.board, debug=True)
            self.N = 1
            self.root = True
        else:
            self.root = False

    def search(self, rolls, rand=False):
        """ Do `rolls` rollouts and return the best move, or a random move as
            by the visit distribution. """
        for _ in range(rolls):
            self._rollout()
            if self.debug:
                sortd = sorted(self.children, key=lambda n:n.N, reverse=True)
                print(', '.join(f'{self.board.san(n.move)}: {n.N/self.N:.1%} {float(-n.Q):.2}' for n in sortd[:5]), end=' \r')
        print()
        if rand:
            counts = [node.N for node in self.children]
            return random.choices(self.children, weights=counts)[0].move
        return max(self.children, key = lambda n: n.N).move

    def _rollout(self):
        """ Returns the leaf value relative to the current player of the node. """

        self.N += 1

        if self.board and self.board.is_game_over():
            v = {'1-0':1, '0-1':-1, '1/2-1/2':0}[self.board.result()]
            return v if self.board.turn == chess.WHITE else -v

        # If first visit, expand board
        if self.N == 1:
            # Don't copy the move stack, it just takes up memory.
            self.board = self.parent_board.copy(stack = False)
            self.board.push(self.move)
            self.Q = self.model.eval(self.board)
            return self.Q

        # If second visit, expand children
        if self.N == 2:
            for p, move in self.model.predict(self.board, debug=self.root):
                if self.board.is_legal(move):
                    self.children.append(Node(self.board, move, p, self.model))
            if not self.children:
                print('Warning: Predictor gave only illegal moves.')
                for move in self.board.legal_moves:
                    self.children.append(Node(self.board, move, .1, self.model))

        # TODO: Paper says cpuct is not constant, but actually
        # C(s) = log ((1 + N(s) + cbase)/cbase) + cinit
        # though it does stay mostly constant during the just 800 rolls.

        # WHat?? De skriver også
        #  U(s, a) = C(s)P(s, a) sqrt(N(s))/(1 + N(s, a))
        # dvs at nævneren slet ikke er med i kvadratroden...
        # Se https://science.sciencemag.org/content/sci/suppl/2018/12/05/362.6419.1140.DC1/aar6404-Silver-SM.pdf

        # Find best child
        node = max(self.children,
                   key = lambda n: -n.Q + cpuct * n.P * sqrt(self.N) / (1 + n.N))
        # Visit it
        s = -node._rollout()
        # Note that Q is an average in the newer paper.
        self.Q = ((self.N-1)*self.Q + s)/self.N
        # Return the terminal value, not our own
        return s


