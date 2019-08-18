import chess
import math
import sys
import random
import fasttext
from collections import defaultdict
import numpy as np
import pst

class Model:
    def __init__(self, path):
        ft = self.ft = fasttext.load_model(path)
        vectors = (ft.get_output_matrix() @ ft.get_input_matrix().T).T
        rows, cols = vectors.shape
        # Add counts
        vectors = np.hstack([np.ones(rows).reshape(rows,1), vectors])
        # maybe its an occ model?
        self.occ = False
        #vectors[i]
        # Start with bias
        bias = vectors[0]
        # Parse remaining words
        piece_to_vec = defaultdict(lambda: 0)
        castling = {}
        for w, v in zip(ft.words[1:], vectors[1:]):
            sq = getattr(chess, w[:2].upper())
            if w.endswith('-Occ'):
                self.occ = True
                for color in chess.COLORS:
                    for piece_type in chess.PIECE_TYPES:
                        piece_to_vec[piece_type, color, sq] += v
            elif w.endswith('-C'):
                castling[sq] = v
            else:
                piece = chess.Piece.from_symbol(w[2])
                piece_to_vec[piece.piece_type, piece.color, sq] += v

        # Convert to two-colours
        piece_to_vec2 = {}
        for (piece_type, color, sq), v in piece_to_vec.items():
            inv = piece_to_vec[piece_type, not color, chess.square_mirror(sq)]
            piece_to_vec2[piece_type, color, sq] = np.vstack([v, inv])

        self.bias = np.vstack([bias, bias])
        self.piece_to_vec = piece_to_vec2
        self.castling = {sq: np.vstack([v, castling[chess.square_mirror(sq)]])
                         for sq, v in castling.items()}

        # Parse labels
        self.moves = [chess.Move.from_uci(label_uci[len('__label__'):]) for label_uci in ft.labels]
        self.move_to_id = {move: i for i, move in enumerate(self.moves)}

    def old_find_moves(self, board, n_labels=20, occ=False):
        """ Returns a list of up to `n_labels` (move, prob) tuples, restricted
            to legal moves.  Probabilities may not sum to 1. """

        if board.turn == chess.BLACK:
            white_moves = self.old_find_moves(board.mirror(), n_labels, occ)
            return [(p, mirror_move(m)) for p, m in white_moves]

        pos = ' '.join(board_to_words(board, occ=occ))
        labels, probs = self.ft.predict(pos, n_labels)
        labels = [l[len('__label__'):] for l in labels]

        return [(p, m) for m, p in zip(map(chess.Move.from_uci, labels), probs)
                if board.is_legal(m)]

    def eval(self, vec, board):
        """ Returns a single score relative to board.turn """

        v = {'1-0': 1, '0-1': -1, '1/2-1/2': 0, '*': None}[board.result()]
        if v is not None:
            return (v if board.turn == chess.WHITE else -v), True

        # We first calculate the value relative to white
        res = 0
        for square, piece in board.piece_map().items():
            p = piece.symbol()
            if p.isupper():
                res += pst.piece[p] + pst.pst[p][63 - square]
            else:
                p = p.upper()
                res -= pst.piece[p] + pst.pst[p][square]
        # Normalize in [-1, 1]
        res = cp_to_win(res)
        # Then flip it to the current player
        return (res if board.turn == chess.WHITE else -res), False

    def get_top_k(self, vec, k):
        for i in np.argpartition(vec, -k)[-k:]:
            yield vec[i], self.moves[i]

    def from_scratch(self, board, debug=False):
        ''' Just for testing that the gradual method works. '''
        vec = self.bias.copy()
        for s, p in board.piece_map().items():
            vec += self.piece_to_vec[p.piece_type, p.color, s]
        for sq in [chess.H1, chess.H8, chess.A1, chess.A8]:
            if board.castling_rights & chess.BB_SQUARES[sq]:
                vec += self.castling[sq]

        if debug:
            v1 = self.ft.get_sentence_vector(' '.join(board_to_words(board, occ=self.occ)))
            v2 = self.ft.get_sentence_vector(' '.join(board_to_words(board.mirror(), occ=self.occ)))
            sv = (self.ft.get_output_matrix() @ np.vstack([v1,v2]).T).T
            n = vec[0,0]
            v = vec[:,1:]
            if not np.allclose(sv, v/n, atol=1e-5, rtol=1e-2):
                print(sv)
                print(v/n)
                print(np.max(np.abs(sv-vec/n)))
                print(np.max(sv/(v/n)), np.min(sv/(v/n)))
                assert False

        return vec

    def apply(self, vec, board, move):
        """ Should be called prior to pushing move to board.
            Applies the move to the vector. """

        # Remove from square.
        piece_type = board.piece_type_at(move.from_square)
        color = board.turn
        vec -= self.piece_to_vec[piece_type, color, move.from_square]

        # Update castling rights.
        old_castling_rights = board.clean_castling_rights()
        new_castling_rights = old_castling_rights & ~chess.BB_SQUARES[move.to_square] & ~chess.BB_SQUARES[move.from_square]
        if piece_type == chess.KING:
            new_castling_rights &= ~chess.BB_RANK_1 if color else ~chess.BB_RANK_8
        # Castling rights can only have been removed
        for sq in chess.scan_forward(old_castling_rights ^ new_castling_rights):
            vec -= self.castling[sq]

        # Remove pawns captured en passant.
        if piece_type == chess.PAWN and move.to_square == board.ep_square:
            down = -8 if board.turn == chess.WHITE else 8
            capture_square = board.ep_square + down
            vec -= self.piece_to_vec[chess.PAWN, not board.turn, capture_square]

        # Move rook during castling.
        if piece_type == chess.KING:
            if move.from_square == chess.E1:
                if move.to_square == chess.G1:
                    vec -= self.piece_to_vec[chess.ROOK, color, chess.H1]
                    vec += self.piece_to_vec[chess.ROOK, color, chess.F1]
                if move.to_square == chess.C1:
                    vec -= self.piece_to_vec[chess.ROOK, color, chess.A1]
                    vec += self.piece_to_vec[chess.ROOK, color, chess.D1]
            if move.from_square == chess.E8:
                if move.to_square == chess.G8:
                    vec -= self.piece_to_vec[chess.ROOK, color, chess.H8]
                    vec += self.piece_to_vec[chess.ROOK, color, chess.F8]
                if move.to_square == chess.C8:
                    vec -= self.piece_to_vec[chess.ROOK, color, chess.A8]
                    vec += self.piece_to_vec[chess.ROOK, color, chess.D8]

        # Capture
        captured_piece_type = board.piece_type_at(move.to_square)
        if captured_piece_type:
            vec -= self.piece_to_vec[captured_piece_type, not color, move.to_square]

        # Put the piece on the target square.
        vec += self.piece_to_vec[move.promotion or piece_type, color, move.to_square]
        return vec

def win_to_cp(win):
    return math.tan(win*math.pi/2)*100

def cp_to_win(cp):
    return math.atan(cp/100)*2/math.pi

def board_to_words(board, occ=False):
    for s, p in board.piece_map().items():
        yield f'{chess.SQUARE_NAMES[s]}{p.symbol()}'
    if board.castling_rights & chess.BB_H1:
        yield 'H1-C'
    if board.castling_rights & chess.BB_H8:
        yield 'H8-C'
    if board.castling_rights & chess.BB_A1:
        yield 'A1-C'
    if board.castling_rights & chess.BB_A8:
        yield 'A8-C'
    if occ:
        for square in chess.scan_forward(board.occupied):
            yield f'{chess.SQUARE_NAMES[square]}-Occ'

# TODO: Consider the following code used by python-chess to compue transposition-keys.
#       Should we use something similar to compress board_to_words?
# def _transposition_key(self) -> Hashable:
#     return (self.pawns, self.knights, self.bishops, self.rooks,
#             self.queens, self.kings,
#             self.occupied_co[WHITE], self.occupied_co[BLACK],
#             self.turn, self.clean_castling_rights(),
#             self.ep_square if self.has_legal_en_passant() else None)

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


class Model_Old:
    def __init__(self, path, occ):
        self.model = fasttext.load_model(path)
        self.occ = occ

    def find_move(self, board, max_labels=20,
                  pick_random=False, debug=True, flipped=False):
        # We always predict form white's perspective
        if board.turn == chess.BLACK:
            return mirror_move(self.find_move(
                board.mirror(), max_labels, pick_random, debug, flipped=True))

        pos = ' '.join(board_to_words(board, occ=self.occ))
        # Keep predicting more labels until a legal one comes up
        for k in range(10, max_labels, 5):
            ps_mvs = self.find_moves(board, max_labels, debug, flipped)
            if not ps_mvs:
                continue

            if pick_random:
                # Return move by probability distribution
                ps, mvs = zip(*ps_mvs)
                return random.choices(mvs, weights=ps)[0]
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
            white_moves = self.find_moves(
                board.mirror(), n_labels, debug, flipped=True)
            return [(p, mirror_move(m)) for p, m in white_moves]

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

        return [(p, m) for m, p in zip(map(chess.Move.from_uci, labels), probs)
                if board.is_legal(m)]
