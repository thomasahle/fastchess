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


class FJL:
    def __init__(self, d, m=1000):
        self.sig = np.random.randint(-1, 1, d)
        self.sam = np.random.randint(0, d//2+1, m//2+1)
    def __matmul__(self, v):
        # FJL is a matrix, dammit
        return np.fft.irfft(np.fft.rfft(self.sig * v)[self.sam])


def tensor_sketch(vec, sketches):
    res = vec
    for sketch in sketches:
        outer = np.einsum('i,j->ij', vec, res).flatten()
        res = sketch @ outer
        #print(outer.shape, vec.shape, res.shape)
        #print(np.linalg.norm(vec), np.linalg.norm(res), np.linalg.norm(outer))
    return res



class ExampleHandler:
    def __init__(self, to_path):
        self.to_path = to_path
        self.boards = []
        self.moves = []
        self.scores = []
        self.sketches = [FJL((6*2*64)**2, 10000)
            #, FJL(6*2*64*1000, 1000)
            ]

        self.move_model = sklearn.linear_model.SGDClassifier(loss='log', n_jobs=8
                )
                #, max_iter=100, tol=.01)

    def add(self, example):
        vec_board, move, score = example
        self.boards.append(tensor_sketch(vec_board, self.sketches))
        self.moves.append(move)
        self.scores.append(score)

        #if len(self.boards) % 1000 = 999:
            #print('Partially fitting model')

    def done(self):
        print('Caching data to games.cached')
        joblib.dump((self.boards, self.moves, self.scores), 'games.cached')

        n = len(self.boards)
        print(f'Got {n} examples')
        p = int(n*.8)

        print('Training move model')
        #move_clf = self.move_model.partial_fit(self.boards[:p], self.moves[:p], classes=range(64**2))
        move_clf = self.move_model.fit(self.boards[:p], self.moves[:p]
                #, classes=range(64**2)
                )
        test = move_clf.score(self.boards[p:], self.moves[p:])
        #clf = sklearn.linear_model.LogisticRegression(
                #solver='saga', multi_class='auto', verbose=1)
        print(f'Test score: {test}')

        print('Training score model.')
        model = sklearn.linear_model.LinearRegression(n_jobs=8)
        score_clf = model.fit(self.boards[:p], self.scores[:p])
        test = score_clf.score(self.boards[p:], self.scores[p:])
        print(f'Test score: {test}')

        joblib.dump(Model(move_clf, score_clf, self.sketches), self.to_path)
        print(f'Saved model as {self.to_path}')

def train_saved():
    eh = ExampleHandler('out.model')
    eh.boards, eh.moves, eh.scores = joblib.load('games.cached')
    eh.done()

if __name__ == '__main__':
    train_saved()

class Model:
    def __init__(self, move_clf, score_clf, sketches):
        self.move_clf = move_clf
        self.score_clf = score_clf
        self.sketches = sketches

    def find_move(self, board, debug=False):
        # Should we flip the board to make sure it always gets it from white?
        vec_board = board_to_vec(board if board.turn == chess.WHITE else board.mirror())
        vec_board = tensor_sketch(vec_board, self.sketches)
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




