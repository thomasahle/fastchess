import fastText
import chess
import fastchess
import re
import sys
import random

def find_move(model, board, max_labels=100):
    pos = ' '.join(fastchess.board_to_words2(board))
    for k in range(10, max_labels, 5):
        labels, probs = model.predict(pos, k)
        print(labels, probs)
        labels = [l[len('__label__'):] for l in labels]
        scores, tos, frs, mvs = [], [], [], []
        for l, p in zip(labels, probs):
            if l.isdigit() or l[0]=='-':
                scores.append(int(l))
            elif l[:2] == 't_':
                tos.append((p, l[2:]))
            elif l[:2] == 'f_':
                frs.append((p, l[2:]))
            else:
                mvs.append((p, l))
        for p1, f in frs:
            for p2, t in tos:
                mvs.append((p1*p2, f+t))
        mvs.sort(reverse=True)
        score = scores[0] if scores else 0
        for p, m in mvs:
            move = chess.Move.from_uci(m)
            if move in board.legal_moves:
                return move, p, score
    print('Warninng: Unable to find a legal move in first {} labels.'
          .format(max_labels))
    return random.choice(list(board.legal_moves)), 0, 0

def user_move(board):
    uci = ''
    while not re.match('[a-h][1-8]'*2, uci):
        uci = input('Your move: ')
    move = chess.Move.from_uci(uci)
    if move in board.legal_moves:
        return move
    print('Illegal move')
    return user_move(board)

def play(model, model_color):
    board = chess.Board()
    while not board.is_game_over():
        if model_color == board.turn:
            move, p, score = find_move(model, board)
            print('My move: uci={} p={:.5} score={}'
                  .format(move.uci(), p, score))
        else:
            print('\n'.join(str(i+1)+' '+line for i, line in enumerate(str(board.mirror()).split('\n'))))
            print('  a b c d e f g h')
            move = user_move(board)
        board.push(move)

def main():
    path = sys.argv[1]
    model = fastText.load_model(path)
    play(model, chess.WHITE)

if __name__ == '__main__':
    main()
