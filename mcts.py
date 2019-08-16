import chess
import numpy as np
import math
from math import sqrt
import random
import fastchess
import pst

class Model:
    def __init__(self, fasttext_model, use_cache=False, policy_softmax_temp=1, formula=0):
        self.fc = fasttext_model
        self.cpuct = 2
        self.cache = {}
        self.use_cache = use_cache
        self.policy_softmax_temp = policy_softmax_temp
        self.formula = formula

    def eval(self, board, debug=False):
        """ Returns a single score relative to board.turn """
        # If game over, just stop
        if board.is_game_over():
            v = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[board.result()]
            return v if board.turn == chess.WHITE else -v

        # We first calculate the value relative to white
        res = 0
        for square, piece in board.piece_map().items():
            p = piece.symbol()
            if p.isupper():
                res += pst.piece[p] + pst.pst[p][63 - square]
            else:
                p = p.upper()
                res -= pst.piece[p] + pst.pst[p][square]
        if debug:
            print('Pre norm score:', res)
        # Normalize in [-1, 1]
        res = math.atan(res / 100) / math.pi * 2
        if debug:
            print('Post norm score:', res)
        # Then flip it to the current player
        return res if board.turn == chess.WHITE else -res

    def predict(self, board, n=40, debug=False):
        """ Returns list of `n` (prob, move) legal pairs """
        if self.use_cache:
            key = board._transposition_key()
            cached = self.cache.get(key)
            if cached:
                return cached
        pre = {m: p for p, m in self.fc.find_moves(
            board, n, debug=debug, flipped=False)}
        res = []
        for m in board.generate_legal_moves():
            cap = board.is_capture(m)
            # TODO: There might be a faster way, inspired by the is_into_check method.
            board.push(m)
            chk = board.is_check()
            board.pop()
            # Hack: We make sure that checks and captures are always included,
            # and that no move has a completely non-existent prior.
            p = max(pre.get(m, 0)**1/self.policy_softmax_temp, .01, .1 * int(chk or cap))
            res.append((p, m))
        psum = sum(p for p, _ in res)
        res = [(p / psum, m) for p, m in res]
        if self.use_cache:
            self.cache[key] = res
        return res


class Node:
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, parent_board, move, prior, model, debug=False):
        """ Make a new game node representing the state of pushing `move` to `parent_board`.
            If `move` is None, the node is assumed to be a root node at `parent_board`. """
        self.children = []
        self.parent_board = parent_board
        self.move = move
        self.board = None  # We expand this as the node is visited
        self.P = prior
        self.Q = .99  # Q, avg reward of node
        self.N = 0  # N, total visit count for node
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
        return max(self.children, key=lambda n: n.N).move

    def ucb(self, n):
        # https://colab.research.google.com/drive/14v45o1xbfrBz0sG3mHbqFtYz_IrQHLTg#scrollTo=1VeRCpCSaHe3
        ratio = n.P * self.N / (1 + n.N)
        # AZ
        if self.formula == 0:
            return -n.Q + ratio * (math.log((self.N + 19652 + 1)/19652) + 1.25) / self.N**.5
        # LZ old
        if self.formula == 1:
            return -n.Q + ratio * 0.8 / self.N**.5
        # Old
        if self.formula == 2:
            return -n.Q + ratio * 3 / self.N**.5
        # LZ new
        if self.formula == 3:
            return -n.Q + ratio * ((.5 * math.log(.015 * self.N + 1.7)) / self.N)**.5
        # UCB
        if self.formula == 4:
            return -n.Q + ratio * (2*math.log(self.N) / (n.N + 1))**.5

    def rollout(self):
        """ Returns the leaf value relative to the current player of the node. """

        self.N += 1

        # TODO: This repeated is_game_over is actually quite expensive
        if self.board and self.board.is_game_over():
            # If board is set, Q should already be the evaluation, which includes
            # checkmate/stalemate.
            return self.Q

        # If first visit, expand board
        if self.N == 1:
            # Don't copy the entire move stack, it just takes up memory.
            # We do need some though, to prevent repetition draws.
            # Half move cluck is copied separately
            self.board = self.parent_board.copy(stack=8)
            self.board.push(self.move)
            self.Q = self.model.eval(self.board, debug=self.debug)
            return self.Q

        # If second visit, expand children
        if self.N == 2:
            for p, move in self.model.predict(self.board, debug=self.debug):
                if self.board.is_legal(move):
                    self.children.append(Node(self.board, move, p, self.model))

        # Identify losses, just an optimization for mates
        # if all(n.Q == 1 for n in self.children):
        #    self.Q = -1
        #    return -1

        # Find best child (small optimization, since this is actually a bottle neck)
        # TODO: Even now, this is still pretty slow
        # _, node = max((-n.Q + CPUCT * n.P * sqrtN / (1 + n.N), n) for n in self.children)
        # is better, but it fails when two nodes have the same score...
        #sqrtN = self.N**.5
        #node = max(self.children,
                   #key=lambda n: -n.Q + self.model.cpuct * n.P * sqrtN / (1 + n.N))
        node = max(self.children, key=self.ucb)

        # Visit it
        s = -node.rollout()
        # Identify victories
        # if s == 1:
        #    self.Q = 1
        #    return 1
        self.Q = ((self.N - 1) * self.Q + s) / self.N
        # Propagate the value further up the tree
        return s

