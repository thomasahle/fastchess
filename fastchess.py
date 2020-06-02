import chess
import math
import sys
import random
import fasttext
from collections import defaultdict
import numpy as np
import pst


def win_to_cp(win):
    ''' Used because uci interface requires cp scores. '''
    return pst.from_win(win)


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


EVAL_INDEX = 0
COUNT_INDEX = 1


class Model:
    def __init__(self, path):
        ft = self.ft = fasttext.load_model(path)
        vectors = (ft.get_output_matrix() @ ft.get_input_matrix().T).T
        rows, _cols = vectors.shape
        # Add counts and evals
        vectors = np.hstack([
            np.ones(rows).reshape(rows, 1),
            vectors])
        # maybe its an occ model?
        self.occ = False

        # Start with bias. No bias for eval.
        bias = np.hstack([[0], vectors[0]])

        # Parse remaining words
        piece_to_vec = defaultdict(lambda: 0)
        castling = {}
        for w, v in zip(ft.words[1:], vectors[1:]):
            sq = getattr(chess, w[:2].upper())
            if w.endswith('-Occ'):
                self.occ = True
                for color in chess.COLORS:
                    for piece_type in chess.PIECE_TYPES:
                        piece_to_vec[piece_type, color, sq] += np.hstack([[0], v])
            elif w.endswith('-C'):
                e = pst.castling[sq]
                castling[sq] = np.hstack([[e], v])
            else:
                p = chess.Piece.from_symbol(w[2])
                e = pst.piece[p.piece_type-1] * (1 if p.color else -1)
                e += pst.pst[0 if p.color else 1][p.piece_type-1][sq]
                #print(w[2], p, e)
                piece_to_vec[p.piece_type, p.color, sq] += np.hstack([[e], v])

        # Convert to two-colours
        # We keep a record of the board from both perspectives
        piece_to_vec2 = {}
        for (piece_type, color, sq), v in piece_to_vec.items():
            inv = piece_to_vec[piece_type, not color, chess.square_mirror(sq)]
            piece_to_vec2[piece_type, color, sq] = np.vstack([v, inv])

        self.bias = np.vstack([bias, bias])
        self.piece_to_vec = piece_to_vec2
        self.castling = {sq: np.vstack([v, castling[chess.square_mirror(sq)]])
                         for sq, v in castling.items()}

        # Parse labels
        self.moves = [chess.Move.from_uci(label_uci[len('__label__'):])
                      for label_uci in ft.labels]

        # Adding 2 to the move ids, since the first entry will be the count,
        # and the second entry will be the evaluation
        self.move_to_id = {move: i + 2 for i, move in enumerate(self.moves)}

    def get_eval(self, vec, board, debug=False):
        """ Returns a single score relative to board.turn """

        cp = vec[1 - int(board.turn), EVAL_INDEX]

        if debug:
            assert vec[0, EVAL_INDEX] == -vec[1, EVAL_INDEX]
            win = pst.to_win(cp)
            fs = self._eval_from_scratch(vec, board)
            assert np.allclose(win, fs)

        # Features that don't require incremental updating
        if board.is_check():
            cp += pst.check
        cp += pst.turn

        #print(board)
        #print(cp)

        return pst.to_win(cp)

    def get_top_k(self, vec, k):
        for i in np.argpartition(vec, -k)[-k:]:
            yield vec[i], self.moves[i]

    # TODO: Maybe we should just subclass chess.Board like in feeks:
    # https://github.com/flok99/feeks/blob/master/board.py
    def apply(self, vec, board, move):
        """ Should be called prior to pushing move to board.
            Applies the move to the vector. """

        # Remove from square.
        piece_type = board.piece_type_at(move.from_square)
        color = board.turn
        vec -= self.piece_to_vec[piece_type, color, move.from_square]

        # Update castling rights.
        old_castling_rights = board.clean_castling_rights()
        new_castling_rights = old_castling_rights & ~chess.BB_SQUARES[
            move.to_square] & ~chess.BB_SQUARES[move.from_square]
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

    def get_clean_moves(self, board, vec, legal_t=1, cap_t=2, chk_t=2, debug=False):
        ''' Returns a list of (prior, move) pairs containing all legal moves. '''
        moves = []
        scores = []
        vec = vec[1 - int(board.turn)]

        if debug:
            vec1 = self.from_scratch(board, debug)[1 - int(board.turn)]
            if not np.allclose(vec, vec1, atol=1e-5, rtol=1e-2):
                print(board)
                print(vec1)
                print(vec)
                assert False

        # Filter out illegal moves.
        # Another approach is to use top_k to get the moves and simply trust
        # that they are legal.
        # self.model.top_k(self.vec)

        for m in board.legal_moves:
            moves.append(m)
            prior = vec[self.move_to_id[m if board.turn else mirror_move(m)]]
            scores.append(prior)

        # A fast text model is normalized.
        # We keep the word count in the first entry.
        n = vec[COUNT_INDEX]
        scores = np.array(scores) / n

        for i, m in enumerate(moves):
            prior = scores[i]
            prior = max(prior, legal_t)

            # Hack: We make sure that checks and captures are always included,
            # and that no move has a completely non-existent prior.
            # Add some bonus for being a legal move and check or cap.
            # These are basically move extensions, like in classical engines.
            # Maybe other extensions would be useful too, like passed pawn or
            # recapture extensions: https://www.chessprogramming.org/Extensions
            if cap_t > prior and board.is_capture(m):
                prior = cap_t

            # TODO: There might be a faster way, inspired by the is_into_check method.
            # or _attackers_mask. Some sort of pseudo-is-check should be sufficient.
            if chk_t > prior:
                board.push(m)
                if board.is_check():
                    prior = chk_t
                board.pop()

            scores[i] = prior

        scores = np.exp(scores - np.max(scores))
        scores /= np.sum(scores)
        return zip(scores, moves)

    def _eval_from_scratch(self, vec, board):
        # We first calculate the value relative to white
        res = 0
        for s, p in board.piece_map().items():
            e = pst.piece[p.piece_type-1] * (1 if p.color else -1)
            e += pst.pst[0 if p.color else 1][p.piece_type-1][s]
            res += e
        for sq in [chess.A1, chess.A8, chess.H1, chess.H8]:
            if board.castling_rights & sq:
                res += pst.castling[sq]
        # Normalize in [-1, 1]
        res = pst.to_win(res)
        # Then flip it to the current player
        return res if board.turn == chess.WHITE else -res

    def from_scratch(self, board, debug=False):
        ''' Just for testing that the gradual method works. '''
        vec = self.bias.copy()
        for s, p in board.piece_map().items():
            vec += self.piece_to_vec[p.piece_type, p.color, s]
        for sq in [chess.H1, chess.H8, chess.A1, chess.A8]:
            if board.castling_rights & chess.BB_SQUARES[sq]:
                vec += self.castling[sq]

        if debug:
            v1 = self.ft.get_sentence_vector(
                ' '.join(board_to_words(board, occ=self.occ)))
            v2 = self.ft.get_sentence_vector(
                ' '.join(
                    board_to_words(
                        board.mirror(),
                        occ=self.occ)))
            sv = (self.ft.get_output_matrix() @ np.vstack([v1, v2]).T).T
            n = vec[0, 1]
            v = vec[:, 2:]
            if not np.allclose(sv, v / n, atol=1e-5, rtol=1e-2):
                print(sv)
                print(v / n)
                print(np.max(np.abs(sv - vec / n)))
                print(np.max(sv / (v / n)), np.min(sv / (v / n)))
                assert False

        return vec


