import chess, chess.uci
import math
import sys
import random
import numpy as np
import sklearn, sklearn.linear_model
from sklearn.externals import joblib


def board_to_vec(board):
    ar = []
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            ar.append(int(board.pieces(piece_type,color)))
    packed = np.unpackbits(np.array(ar).view(np.uint8))
    return packed


def move_to_vec(move):
    return move.from_square + move.to_square*64


def pscore(score):
    if score.cp is not None:
        cp = score.cp
        score = 2/(1 + 10**(-cp/400)) - 1
        #score = cp
        return score
    elif score.mate < 0:
        return -1
    elif score.mate > 0:
        return 1


def prepare_example(board, move, score):
    if board.turn == chess.WHITE:
        vec_board = board_to_vec(board)
        vec_move = move.from_square + move.to_square*64
    else:
        vec_board = board_to_vec(board.mirror())
        vec_move = chess.square_mirror(move.from_square) + chess.square_mirror(move.to_square)*64
    #print(score, pscore(score))
    return vec_board, vec_move, pscore(score)


def fjl(d, m=100):
    sig = np.random.randint(-1, 1, len(outer))
    def func(vec):
        return np.fft.irfft(np.random.choice(np.fft.rfft(sig * outer), m//2+1))
    return func


def tensor_sketch(vec, sketches):
    res = vec
    for sketch in sketches:
        outer = np.einsum('i,j->ij', vec, res).flatten()
        res = sketch(outer)
    return res


def train(path):
    pass


class ExampleHandler:
    def __init__(self):
        self.boards = []
        self.moves = []
        self.scores = []

    def add(self, example):
        vec_board, vec_move, score = example
        self.boards.append(vec_board)
        self.moves.append(vec_move)
        self.scores.append(score)

    def done(self):
        n = len(self.boards)
        print(f'Got {n} examples')
        p = int(n*.8)

        print('Training move model')
        model = sklearn.linear_model.SGDClassifier(loss='log', n_jobs=8, max_iter=10, tol=.1)
        #clf = sklearn.linear_model.LogisticRegression(
                #solver='saga', multi_class='auto', verbose=1)
        move_clf = model.partial_fit(self.boards[:p], self.moves[:p], classes=range(64**2))
        test = move_clf.score(self.boards[p:], self.moves[p:])
        print(f'Test score: {test}')

        print('Training score model.')
        model = sklearn.linear_model.LinearRegression(n_jobs=8)
        score_clf = model.fit(self.boards[:p], self.scores[:p])
        test = score_clf.score(self.boards[p:], self.scores[p:])
        print(f'Test score: {test}')

        joblib.dump((move_clf, score_clf), 'tensor.model')
        print('Saved model as tensor.model')


class Model:
    def __init__(self, path):
        self.move_clf, self.score_clf = joblib.load('tensor.model')

    def find_move(self, board, debug=False):
        # Should we flip the board to make sure it always gets it from white?
        vec_board = board_to_vec(board if board.turn == chess.WHITE else board.mirror())
        probs = self.move_clf.predict_proba([vec_board])[0]
        score = self.score_clf.predict([vec_board])[0]
        for n, (mp, from_to) in enumerate(sorted((-p,ft) for ft,p in enumerate(probs))):
            to_square, from_square = divmod(from_to, 64)
            if board.turn == chess.BLACK:
                from_square = chess.square_mirror(from_square)
                to_square = chess.square_mirror(to_square)
            move = chess.Move(from_square, to_square)
            if move in board.legal_moves:
                print(f'Choice move {n}, p={-mp}')
                return move, score




