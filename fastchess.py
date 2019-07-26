import chess, chess.uci
import math
import sys
import random
import fasttext

#CONVOLUTIONS = [(1,1),(2,2),(3,2),(2,3)]
CONVOLUTIONS = [(1,1)]

def board_to_words(board):
    piece_map = board.piece_map()
    word = []
    for w, h in CONVOLUTIONS:
        for f1 in range(-1, 10-w):
            for r1 in range(-1, 10-h):
                del word[:]
                word.append('_'.join(map(str,[w,h,f1,r1])))
                any_pieces = False
                for f in range(f1, f1+w):
                    for r in range(r1, r1+h):
                        if f < 0 or f > 7 or r < 0 or r > 7:
                            word.append('x')
                        else:
                            s = chess.square(f,r)
                            piece = piece_map.get(s, None)
                            if piece is None:
                                word.append('-')
                            else:
                                any_pieces = True
                                word.append(piece.symbol())
                if any_pieces:
                    yield ''.join(word)


def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      move.promotion)


def prepare_example(board, move):
    if board.turn == chess.WHITE:
        string = ' '.join(board_to_words(board))
        uci_move = move.uci()
    else:
        string = ' '.join(board_to_words(board.mirror()))
        uci_move = mirror_move(move).uci()
    return f'{string} __label__{uci_move}'


class ExampleHandler:
    def __init__(self):
        pass
    def add(self, example):
        print(example)
    def done(self):
        pass


class Model:
    def __init__(self, path):
        self.model = fasttext.load_model(path)

    def find_move(self, board, max_labels=20, pick_random=False, debug=True, flipped=False):
        if board.turn == chess.BLACK:
            return mirror_move(self.find_move(
                    board.mirror(), max_labels, pick_random, debug, flipped=True))

        #pos = ' '.join(board_to_words(board if board.turn == chess.WHITE else board.mirror()))
        pos = ' '.join(board_to_words(board))
        for k in range(10, max_labels, 5):
            labels, probs = self.model.predict(pos, k)
            labels = [l[len('__label__'):] for l in labels]

            if debug:
                ucis = [chess.Move.from_uci(l) for l in labels]
                db = board
                if flipped:
                    ucis = [mirror_move(uci) for uci in ucis]
                    db = board.mirror()
                top_list = []
                for uci, p in zip(ucis, probs):
                    if db.is_legal(uci):
                        san = db.san(uci)
                        tag = f'{p:.1%}'
                    else:
                        san = uci.uci()
                        tag = f'illegal, {p:.1%}'
                    top_list.append(f'{san} ({tag})')
                print('Top moves:', ', '.join(top_list))

            ps_mvs = [(p,m) for m, p in zip(map(chess.Move.from_uci, labels), probs) if board.is_legal(m)]
            if not ps_mvs:
                continue

            if pick_random:
                # Return move by probability distribution
                ps, mvs = zip(*ps_mvs)
                return random.choices(mvs, weights = ps)[0]
            else:
                # Return best legal move
                p, mv = max(ps_mvs)
                return mv

        if debug:
            print('Warninng: Unable to find a legal move in first {} labels.'
                  .format(max_labels))

        return random.choice(list(board.legal_moves))

