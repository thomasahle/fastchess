import chess
import re
import fastchess
import argparse
import random
import time

def get_user_move(board):
    # Get well-formated move
    move = None
    while move is None:
        san_option = random.choice([board.san(m) for m in board.legal_moves])
        uci_option = random.choice([m.uci() for m in board.legal_moves])
        uci = input(f'Your move (e.g. {san_option} or {uci_option}): ')
        try:
            move = board.parse_san(uci)
        except ValueError:
            pass
        if re.match('^[a-h][1-8]'*2+'[qrkb]?$', uci):
            move = chess.Move.from_uci(uci)

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
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙'}
    board_str = []
    for i, p in enumerate(str(board)):
        if p == '.' and (i//2 + i//16) % 2 == 0: board_str.append('□')
        if p == '.' and (i//2 + i//16) % 2 == 1: board_str.append('·')
        if p != '.': board_str.append(uni_pieces.get(p,p))
    board_str = ''.join(board_str).split('\n')
    if perspective == chess.WHITE:
        print('\n'.join(f'{8-i} {line}'
                for i, line in enumerate(board_str)))
        print('  a b c d e f g h\n')
    else:
        print('\n'.join(f'{1+i} {line[::-1]}'
                for i, line in enumerate(board_str[::-1])))
        print('  h g f e d c b a\n')


def self_play(model, rand=False, debug=False):
    board = chess.Board()
    # Just a test that promotion does work.
    #board = chess.Board('8/P7/5kp1/4p2p/2p5/4P2P/6PK/2q5 w KQkq - 0 1')

    while not board.is_game_over():
        print_unicode_board(board, perspective = chess.WHITE)
        move = model.find_move(board, debug=debug, pick_random=rand)
        print(f' My move: {board.san(move)}')
        board.push(move)

    # Print status
    print_unicode_board(board)
    print('Result:', board.result(), 'Status:', board.status())

def play(model, rand=False, debug=False):
    user_color = get_user_color()
    board = chess.Board()

    try:
        while not board.is_game_over():
            print_unicode_board(board, perspective = user_color)
            #import tensorsketch
            #print(board, board.mirror())
            #print(tensorsketch.board_to_vec(board))
            if user_color == board.turn:
                move = get_user_move(board)
            else:
                time.sleep(.3)
                move = model.find_move(board, debug=debug, pick_random=rand)
                print(f'My move: {board.san(move)}')
            board.push(move)

        # Print status
        print_unicode_board(board, perspective = user_color)
        print('Result:', board.result())

    except KeyboardInterrupt:
        print('\nGoodbye!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Location of fasttext model to use')
    parser.add_argument('-selfplay', action='store_true', help='Play against itself')
    parser.add_argument('-rand', action='store_true', help='Play random moves from predicted distribution')
    parser.add_argument('-debug', action='store_true', help='Print all predicted labels')
    args = parser.parse_args()

    print('Loading model...')
    model = fastchess.Model(args.model_path)
    # from sklearn.externals import joblib
    # model = joblib.load(path)
    if args.selfplay:
        self_play(model, rand=args.rand, debug=args.debug)
    else:
        play(model, rand=args.rand, debug=args.debug)

if __name__ == '__main__':
    main()
