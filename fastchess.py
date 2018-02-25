import chess, chess.uci
import math
import sys
import random

CONVOLUTIONS = [(1,1),(2,2),(3,2),(2,3)]
SCORE_STEPS = 10
MAX_SCORE = 750

def discretize_score(score):
    if score.cp is not None:
        cp = score.cp
        if cp > MAX_SCORE: return SCORE_STEPS
        if cp < -MAX_SCORE: return -SCORE_STEPS
        score = 2/(1 + 10**(-cp/400)) - 1
        return int(score*SCORE_STEPS)
    else:
        return 'm{}'.format(score.mate)

def undiscretize_score(score):
    if score[0] == 'm':
        steps = int(score[1:])
        if steps < 0: return -MAX_SCORE + steps
        if steps > 0: return MAX_SCORE + steps
    else:
        x = int(score)
        if x == SCORE_STEPS: return MAX_SCORE
        if x == -SCORE_STEPS: return -MAX_SCORE
        x /= SCORE_STEPS
        return 400*math.log((1 + x)/(1 - x))/math.log(10)

def board_to_words(board):
    piece_map = board.piece_map()
    for w, h in CONVOLUTIONS:
        for f1 in range(-1, 10-w):
            for r1 in range(-1, 10-h):
                word = ['_'.join(map(str,[w,h,f1,r1]))]
                for f in range(f1, f1+w):
                    for r in range(r1, r1+h):
                        if f < 0 or f > 7 or r < 0 or r > 7:
                            word.append('x')
                        else:
                            s = chess.square(f,r)
                            piece = piece_map.get(s, None)
                            if piece is None:
                                word.append('-')
                            else:
                                word.append(piece.symbol())
                yield ''.join(word)

def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      move.promotion)

def find_move(model, board, max_labels=100, pick_random=False, debug=False):
    if board.turn == chess.BLACK:
        move, score = find_move(model, board.mirror(), max_labels,
                                pick_random, debug)
        return mirror_move(move), -score

    pos = ' '.join(board_to_words(board if board.turn == chess.WHITE
                                        else board.mirror()))
    for k in range(10, max_labels, 5):
        labels, probs = model.predict(pos, k)
        labels = [l[len('__label__'):] for l in labels]
        if debug:
            print(', '.join('{}: {:.5}'.format(l,p) for l,p in zip(labels, probs)), file=sys.stderr)
        scores, tos, frs, mvs = [], [], [], []
        for l, p in zip(labels, probs):
            if l.isdigit() or l[0] in 'm-':
                scores.append((p, undiscretize_score(l)))
            elif l[:2] == 't_':
                tos.append((p, l[2:]))
            elif l[:2] == 'f_':
                frs.append((p, l[2:]))
            else:
                mvs.append((p, l))

        # Make the score a weighted average of predicted scores
        score = int(sum(p*v for p,v in scores)/scores[0][0]) if scores else 0

        # Add combinations of to/from's to the list of possible moves
        for p1, f in frs:
            for p2, t in tos:
                mvs.append((p1*p2, f+t))

        if pick_random:
            # Return any legal predicted move
            uci_moves = (chess.Move.from_uci(m) for _, m in mvs)
            legal_moves = [m for m in uci_moves if m in board.legal_moves]
            return random.choice(legal_moves), score
        else:
            # Return best legal move
            mvs.sort(reverse=True)
            for p, m in mvs:
                move = chess.Move.from_uci(m)
                if move in board.legal_moves:
                    return move, score
    if debug:
        print('Warninng: Unable to find a legal move in first {} labels.'
              .format(max_labels), file=sys.stderr)
    return random.choice(list(board.legal_moves)), 0

