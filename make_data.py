import chess
import chess.uci
import random
import itertools
import multiprocessing
import sys
import argparse
import time
from datetime import timedelta

STOCKFISH_PATH = "/Users/thdy/Repos/fastchess/Stockfish/src/stockfish"

parser = argparse.ArgumentParser(
        description='Make fasttext input for learning chess.')
parser.add_argument('-color', type=str, default='white',
                    help='Learn moves with respect to this color.')
parser.add_argument('-movetime', type=int, default=10,
                    help='Milliseconds per move for stockfish.')
parser.add_argument('-threads', type=int, default=8,
                    help='Threads to use.')
parser.add_argument('-games', type=int, default=8,
                    help='Number of games to play')
parser.add_argument('-splitmove', action='store_const', const=True,
                    default=False, help='Whether to have to/from separate.')
parser.add_argument('-convolutions', type=str, default='1x1,2x2',
                    help='Types of convolutions to use, default=1x1,2x2')
args = parser.parse_args()

OUR_COLOR = chess.WHITE if args.color == 'white' else chess.BLACK
SPLIT_MOVE = args.splitmove
CONVOLUTIONS = [list(map(int,c.split('x'))) for c in args.convolutions.split(',')]
MOVE_TIME = args.movetime
THREADS = args.threads
N_GAMES = args.games

def discretize_score(score):
    if score.cp is not None:
        cp = score.cp
        steps = 10
        if cp > 750: return steps
        if cp < -750: return -steps
        score = 2/(1 + 10**(-cp/400)) - 1
        return int(score*steps)
    else:
        return 'm{}'.format(score.mate)

def make_convolutions(square, width, height):
    # Convolutions are made a bit more complicated due to even sizes
    file_ = chess.square_file(square)
    rank_ = chess.square_rank(square)
    f_ranges, r_ranges = [], []
    for ranges, start, size in ((f_ranges, file_, width),
                                (r_ranges, rank_, height)):
        if size % 2 == 1:
            ranges.append(range(start-size//2, start+size//2+1))
        else:
            ranges.append(range(start-size//2, start+size//2))
            ranges.append(range(start-size//2+1, start+size//2+1))
    for f_range in f_ranges:
        for r_range in r_ranges:
            convolution = []
            for f in f_range:
                for r in r_range:
                    if f < 0 or f > 7 or r < 0 or r > 7:
                        convolution.append(None)
                    else: convolution.append(chess.square(f,r))
            yield convolution

convolutions = {square: [c for w, h in CONVOLUTIONS
                           for c in make_convolutions(square, w, h)]
                for square in chess.SQUARES}

def board_to_words(board):
    piece_map = board.piece_map()
    for square in piece_map.keys():
        for convolution in convolutions[square]:
            word = [str(square)]
            for s in convolution:
                if s is None:
                    word.append('x')
                elif s in piece_map:
                    word.append(piece_map[s].symbol())
                else:
                    word.append('-')
            yield ''.join(word)

def play_game(engine, info_handler):
    board = chess.Board()
    engine.ucinewgame()
    while not board.is_game_over():
        engine.position(board)
        engine.isready()

        # If we starve the engine on time, it may not find a move
        tries = 1
        score = {}
        while 1 not in score:
            move = engine.go(movetime=MOVE_TIME*tries).bestmove
            score = info_handler.info["score"]
            tries += 1

        # Make fasttext line
        if board.turn == OUR_COLOR:
            labels = []
            labels.append(str(discretize_score(score[1])))
            uci_move = move.uci()[:4]
            if SPLIT_MOVE:
                labels.append('f_' + uci_move[:2])
                labels.append('t_' + uci_move[2:])
            else:
                labels.append(uci_move)
            labels = ['__label__' + label for label in labels]
            yield ' '.join(itertools.chain(board_to_words(board), labels))

        # In the beginning of the game, add some randomness
        if 1/board.fullmove_number > random.random():
            move = random.choice(list(board.legal_moves))
        board.push(move)

def run_thread(thread_id, print_lock):
    engine = chess.uci.popen_engine(STOCKFISH_PATH)
    info_handler = chess.uci.InfoHandler()
    engine.info_handlers.append(info_handler)
    engine.uci()
    engine.isready()
    start, last = time.time(), 0
    for i in range(N_GAMES//THREADS):
        if thread_id == 0 and time.time()-last > .5:
            pg = i*THREADS/N_GAMES
            if i == 0: pg += 1/1000000
            etr = (time.time() - start)*(1/pg-1)
            print('Progress: {:.1f}%. Remaining: {}'
                  .format(pg*100, str(timedelta(seconds=int(etr)))),
                  file=sys.stderr)
            last = time.time()
        for line in play_game(engine, info_handler):
            with print_lock:
                print(line)

def main():
    lock = multiprocessing.Lock()
    ps = []
    for i in range(THREADS):
        p = multiprocessing.Process(target=run_thread, args=(i, lock))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

if __name__ == '__main__':
    main()
