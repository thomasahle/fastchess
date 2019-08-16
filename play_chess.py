import chess
import re
import fastchess
import argparse
import random
import time
import mcts
import numpy as np
import time
from controller import MCTS_Controller


parser = argparse.ArgumentParser()
parser.add_argument('model_path', help='Location of fasttext model to use')
parser.add_argument('-selfplay', action='store_true',
                    help='Play against itself')
parser.add_argument('-rand', nargs='?', help='Play random moves from the posterior distribution to the 1/temp power.',
                    metavar='TEMP', const=1, default=0, type=float)
parser.add_argument('-debug', action='store_true',
                    help='Print all predicted labels')
parser.add_argument('-mcts', nargs='?', help='Play stronger (hopefully)',
                    metavar='ROLLS', const=800, default=1, type=int)
parser.add_argument(
    '-pvs', nargs='?', help='Show Principal Variations (when mcts)', const=3, default=0, type=int)
parser.add_argument('-fen', help='Start from given position',
                    default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
parser.add_argument('-occ', action='store_true', help='Add -Occ features')
parser.add_argument('-cache', action='store_true', help='Cache outputs from fasttext')
parser.add_argument('-profile', action='store_true',
                    help='Run profiling. (Only with selfplay)')


def get_user_move(board):
    # Get well-formated move
    move = None
    while move is None:
        san_option = random.choice([board.san(m) for m in board.legal_moves])
        uci_option = random.choice([m.uci() for m in board.legal_moves])
        uci = input(f'Your move (e.g. {san_option} or {uci_option}): ')
        for parse in (board.parse_san, chess.Move.from_uci):
            try:
                move = parse(uci)
            except ValueError:
                pass

    # Check legality
    if move not in board.legal_moves:
        print('Illegal move.')
        return get_user_move(board)

    return move


def get_user_color():
    color = ''
    while color not in ('white', 'black'):
        color = input('Do you want to be white or black? ')
    return chess.WHITE if color == 'white' else chess.BLACK


def print_unicode_board(board, perspective=chess.WHITE):
    """ Prints the position from a given perspective. """
    uni_pieces = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'}
    sc, ec = '\x1b[0;30;107m', '\x1b[0m'
    for r in range(8) if perspective == chess.BLACK else range(7, -1, -1):
        line = [f'{sc} {r+1}']
        for c in range(8) if perspective == chess.WHITE else range(7, -1, -1):
            color = '\x1b[48;5;255m' if (r + c) % 2 == 1 else '\x1b[48;5;253m'
            if board.move_stack:
                if board.move_stack[-1].to_square == 8 * r + c:
                    color = '\x1b[48;5;153m'
                elif board.move_stack[-1].from_square == 8 * r + c:
                    color = '\x1b[48;5;153m'
            piece = board.piece_at(8 * r + c)
            if piece:
                line.append(color + uni_pieces[piece.symbol()])
            else:
                line.append(color + ' ')
        print(' ' + ' '.join(line) + f' {sc} {ec}')
    if perspective == chess.WHITE:
        print(f' {sc}   a b c d e f g h  {ec}\n')
    else:
        print(f' {sc}   h g f e d c b a  {ec}\n')


def play(model, rolls, rand=0, debug=False, board=None, pvs=0, selfplay=False):
    if not selfplay:
        user_color = get_user_color()
    else: user_color = chess.WHITE

    if not board:
        board = chess.Board()

    while not board.is_game_over():
        print_unicode_board(board, perspective=user_color)
        if not selfplay and user_color == board.turn:
            move = get_user_move(board)
        else:
            node, stats = model.find_move(board, min_kldiv=1/rolls, debug=debug, temperature=rand, pvs=pvs)
            move = node.move
            print(f' My move: {board.san(move)}')
        board.push(move)

    # Print status
    print_unicode_board(board, perspective=user_color)
    print('Result:', board.result())


def main():
    args = parser.parse_args()
    if args.debug:
        print('Loading model...')
    fastchess_model = fastchess.Model(args.model_path, occ=args.occ)
    model = MCTS_Controller(fastchess_model, use_cache=args.cache)
    board = chess.Board(args.fen)

    try:
        if args.selfplay:
            if args.profile:
                import cProfile as profile
                profile.runctx(
                    'play(model, rolls=args.mcts, rand=args.rand, debug=args.debug, board=board, pvs=args.pvs, selfplay=True)', globals(), locals())
            else:
                play(model, rolls=args.mcts, rand=args.rand, debug=args.debug, board=board, pvs=args.pvs, selfplay=True)
        else:
            play(model, rolls=args.mcts, rand=args.rand, debug=args.debug, board=board, pvs=args.pvs)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nGoodbye!')


if __name__ == '__main__':
    main()
