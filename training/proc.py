import chess.pgn
import random
from pathlib import Path
import argparse
import fastchess


def binary_encode(board):
    """ Returns the board as a binary vector, for eval prediction purposes. """
    rows = []
    for color in [chess.WHITE, chess.BLACK]:
        for ptype in range(chess.PAWN, chess.KING + 1):
            mask = board.pieces_mask(ptype, color)
            rows.append(list(map(int, bin(mask)[2:].zfill(64))))
    ep = [0] * 64
    if board.ep_square:
        ep[board.ep_square] = 1
    rows.append(ep)
    rows.append([
        int(board.turn),
        int(bool(board.castling_rights & chess.BB_A1)),
        int(bool(board.castling_rights & chess.BB_H1)),
        int(bool(board.castling_rights & chess.BB_A8)),
        int(bool(board.castling_rights & chess.BB_H8)),
        int(board.is_check())
    ])
    return np.concatenate(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', help='glob for pgn files, e.g. **/*.pgn')
    parser.add_argument('-test', help='test out')
    parser.add_argument('-train', help='train out')
    parser.add_argument('-ttsplit', default=.8, help='test train split')
    parser.add_argument('-eval', action='store_true',
                        help='predict eval rather than moves')
    parser.add_argument('-occ', action="store_true")
    args = parser.parse_args()

    # TODO: Consider input features for
    # - castling and ep-rights
    # - past move
    # - occupied squares (makes it easier to play legally)
    # - whether the king is in check
    # - attackers/defenders fr each square

    progress = 0
    last_print = 0
    with open(args.test, 'w') as testfile, open(args.train, 'w') as trainfile:
        for p in Path('.').glob(args.files):
            print('Doing', p)
            with open(p) as file:
                for game in iter(lambda: chess.pgn.read_game(file), None):
                    if progress >= last_print + 100:
                        last_print = progress
                        print(progress, end='\r')
                        testfile.flush()
                        trainfile.flush()

                    for node in game.mainline():
                        # Weirdly, the board associated with the node is the result
                        # of the move, rather than from before the move
                        line = fastchess.prepare_example(
                            node.parent.board(), node.move, 0, only_move=True, occ=args.occ)
                        print(line, file=(
                            trainfile if random.random() < args.ttsplit else testfile))
                        progress += 1

    print('Done!')
