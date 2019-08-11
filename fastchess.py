import chess
import math
import sys
import random
import fasttext

def board_to_words(board, occ=False):
    for s, p in board.piece_map().items():
        yield f'{chess.SQUARE_NAMES[s]}{p.symbol()}'
    if board.castling_rights & chess.BB_H1: yield 'H1-C'
    if board.castling_rights & chess.BB_H8: yield 'H8-C'
    if board.castling_rights & chess.BB_A1: yield 'A1-C'
    if board.castling_rights & chess.BB_A8: yield 'A8-C'
    if occ:
        for square in chess.scan_forward(board.occupied):
            yield f'{chess.SQUARE_NAMES[square]}-Occ'

def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      move.promotion)


def prepare_example(board, move, occ=False):
    if board.turn == chess.WHITE:
        string = ' '.join(board_to_words(board, occ=occ))
        uci_move = move.uci()
    else:
        string = ' '.join(board_to_words(board.mirror(), occ=occ))
        uci_move = mirror_move(move).uci()
    return f'{string} __label__{uci_move}'


class Model:
    def __init__(self, path, occ):
        self.model = fasttext.load_model(path)
        self.occ = occ

    def find_move(self, board, max_labels=20, pick_random=False, debug=True, flipped=False):
        # Keep predicting more labels until a legal one comes up
        if board.turn == chess.BLACK:
            return mirror_move(self.find_move(
                    board.mirror(), max_labels, pick_random, debug, flipped=True))

        pos = ' '.join(board_to_words(board, occ=self.occ))
        for k in range(10, max_labels, 5):
            ps_mvs = self.find_moves(board, max_labels, debug, flipped)
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
            print('Warning: Unable to find a legal move in first {} labels.'
                  .format(max_labels))

        return random.choice(list(board.legal_moves))

    def find_moves(self, board, n_labels=20, debug=True, flipped=False):
        """ Returns a list of up to `n_labels` (move, prob) tuples, restricted
            to legal moves.  Probabilities may not sum to 1. """

        if board.turn == chess.BLACK:
            white_moves = self.find_moves(board.mirror(), n_labels, debug, flipped=True)
            return [(p,mirror_move(m)) for p,m in white_moves]

        pos = ' '.join(board_to_words(board, occ=self.occ))
        labels, probs = self.model.predict(pos, n_labels)
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

        return [(p,m) for m, p in zip(map(chess.Move.from_uci, labels), probs)
                    if board.is_legal(m)]

