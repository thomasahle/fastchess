import fastText
import chess
import fastchess
import re
import sys
import random

def get_user_move(board):
    lines = ['{} {}'.format(8-i, line)
        for i, line in enumerate(str(board).split('\n'))]
    print('\n'.join(lines if board.turn == chess.WHITE else lines[::-1]))
    print('  a b c d e f g h')

    # Get well-formated move
    uci = ''
    while not re.match('[a-h][1-8]'*2+'[qrkb]?', uci):
        uci = input('Your move (e.g. g1f3): ')

    # Check legality
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        print('Illegal move.')
        return get_user_move(board)

    return move

def get_user_color():
    color = ''
    while color not in ('white', 'black'):
        color = input('Do you want to be white or black? ')
    return chess.WHITE if color == 'white' else chess.BLACK

def play(model):
    user_color = get_user_color()
    board = chess.Board()
    while not board.is_game_over():
        if user_color == board.turn:
            move = get_user_move(board)
        else:
            move, score = fastchess.find_move(model, board, debug=True)
            print('My move: uci={} score={}'.format(move.uci(), score))
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
