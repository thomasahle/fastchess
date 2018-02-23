import chess
import chess.uci
import random
import itertools
import multiprocessing
import sys
import argparse
import time
from datetime import timedelta
import fastchess

#STOCKFISH_PATH = "Stockfish/src/stockfish"
STOCKFISH_PATH = "/data2/fc/Stockfish/src/stockfish"

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
parser.add_argument('-convolutions', type=str, default='1x1,2x2',
                    help='Types of convolutions to use, default=1x1,2x2')
args = parser.parse_args()

OUR_COLOR = chess.WHITE if args.color == 'white' else chess.BLACK
#CONVOLUTIONS = [list(map(int,c.split('x'))) for c in args.convolutions.split(',')]
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
            labels.append('__label__' + str(discretize_score(score[1])))
            # We add split moves as well as actual move
            uci_move = move.uci()[:4]
            labels.append('__label__' + 'f_' + uci_move[:2])
            labels.append('__label__' + 't_' + uci_move[2:])
            labels.append('__label__' + uci_move)
            yield ' '.join(itertools.chain(fastchess.board_to_words(board), labels))

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
    if thread_id == 0:
        print('Finishing remaining threads...')

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
