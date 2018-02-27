import fastText
import chess
import fastchess
import re
import sys
import random

def get_user_move(board):
    print_unicode_board(board)

    # Get well-formated move
    move = None
    while move is None:
        uci = input('Your move (e.g. g1f3 or Nf3): ')
        try:
            move = board.parse_san(uci)
        except ValueError:
            pass
        if re.match('[a-h][1-8]'*2+'[qrkb]?', uci):
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

def print_unicode_board(board):
    print()
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙'}
    board_str = []
    for i, p in str(board):
        if p == '.' and i % 2 == 0: board_str.append('•')
        if p == '.' and i % 2 == 1: board_str.append('·')
        if p != '.': board_str.append(uni_pieces.get(p,p))
    board_str = ''.join(board_str).split('\n')
    lines = ['{} {}'.format(8-i, line) for i, line in enumerate(board_str)]
    print('\n'.join(lines if board.turn == chess.WHITE else lines[::-1]))
    print('  a b c d e f g h\n')

def play(model):
    user_color = get_user_color()
    board = chess.Board()
    while not board.is_game_over():
        if user_color == board.turn:
            move = get_user_move(board)
        else:
            move, score = fastchess.find_move(model, board, debug=True)
            print('My move: {} score={}cp'.format(board.san(move), score))
        board.push(move)

    # Print status
    print(board)
    print('Game result:', board.result())

def main():
    print('Loading model...')
    path = sys.argv[1]
    model = fastText.load_model(path)
    play(model)

if __name__ == '__main__':
    main()
