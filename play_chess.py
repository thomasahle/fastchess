import chess
import re
import fastchess
import argparse
import random
import time
import mcts

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
    print()
    uni_pieces = {
            'r':'♜', 'n':'♞', 'b':'♝', 'q':'♛', 'k':'♚', 'p':'♟',
            'R':'♖', 'N':'♘', 'B':'♗', 'Q':'♕', 'K':'♔', 'P':'♙',
            '.': ' ', ' ': ' ', '\n': '\n'}
    board_str = str(board)
    if perspective == chess.BLACK:
        board_str = '\n'.join(line[::-1] for line in board_str.split('\n')[::-1])
    colored = []
    for i, p in enumerate(board_str):
        if (i//2 + i//16) % 2 == 0: colored.append('\x1b[0;30;107m' + uni_pieces[p])
        if (i//2 + i//16) % 2 == 1: colored.append('\x1b[0;30;47m' + uni_pieces[p])
    lines = ''.join(colored).split('\n')
    sc, ec = '\x1b[0;30;107m', '\x1b[0m'
    if perspective == chess.WHITE:
        print('\n'.join(f' {sc} {8-i} {line} {ec}' for i, line in enumerate(lines)))
        print(f' {sc}   a b c d e f g h {ec}\n')
    else:
        print('\n'.join(f' {sc} {1+i} {line} {ec}' for i, line in enumerate(lines)))
        print(f' {sc}   h g f e d c b a {ec}\n')


def self_play(model, rand=False, debug=False):
    board = chess.Board()
    #board = chess.Board('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1')
    # Just a test that promotion does work.
    #board = chess.Board('8/P7/5kp1/4p2p/2p5/4P2P/6PK/2q5 w KQkq - 0 1')

    while not board.is_game_over():
        print_unicode_board(board)
        move = model.find_move(board, debug=debug, pick_random=rand)
        print(f' My move: {board.san(move)}')
        board.push(move)

    # Print status
    print_unicode_board(board)
    print('Result:', board.result(), 'Status:', board.status())

def play(model, rand=False, debug=False, sleep=0):
    user_color = get_user_color()
    board = chess.Board()

    try:
        while not board.is_game_over():
            print_unicode_board(board, perspective = user_color)
            if user_color == board.turn:
                move = get_user_move(board)
            else:
                time.sleep(sleep)
                move = model.find_move(board, debug=debug, pick_random=rand)
                print(f'My move: {board.san(move)}')
            board.push(move)

        # Print status
        print_unicode_board(board, perspective = user_color)
        print('Result:', board.result())

    except KeyboardInterrupt:
        print('\nGoodbye!')

class MCTS_Model:
    def __init__(self, fasttext_model):
        self.model = mcts.Model(fasttext_model)

    def find_move(self, board, debug=False, pick_random=False):
        # TODO: Reuse previous nodes
        root = mcts.Node(board, None, 0, self.model)
        return root.search(rolls = 8000, rand = pick_random)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Location of fasttext model to use')
    parser.add_argument('-selfplay', action='store_true', help='Play against itself')
    parser.add_argument('-rand', action='store_true', help='Play random moves from predicted distribution')
    parser.add_argument('-debug', action='store_true', help='Print all predicted labels')
    parser.add_argument('-mcts', action='store_true', help='Play stronger (hopefully)')
    args = parser.parse_args()

    print('Loading model...')
    model = fastchess.Model(args.model_path)
    if args.mcts:
        model = MCTS_Model(model)
    if args.selfplay:
        self_play(model, rand=args.rand, debug=args.debug)
    else:
        play(model, rand=args.rand, debug=args.debug,
                sleep = .3 if not args.mcts else 0)

if __name__ == '__main__':
    main()

