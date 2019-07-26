import chess
import chess.uci
import random
import itertools
import multiprocessing
import sys
import argparse
import time
from datetime import timedelta

STOCKFISH_PATH = '/usr/local/Cellar/stockfish/10/bin/stockfish'

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
parser.add_argument('-alg', type=str, default='tensor',
                    help='fasttext or tensor')
parser.add_argument('-model', type=str, default=None,
                    help='Fastchess model to play with')
parser.add_argument('-to', type=str,
                    help='Where to store the model')
args = parser.parse_args()

MOVE_TIME = args.movetime
THREADS = args.threads
N_GAMES = args.games

def play_game(engine, info_handler, model, selfplay_model=None):
    board = chess.Board()
    engine.ucinewgame()
    while not board.is_game_over() and board.fullmove_number < 60:
        engine.position(board)
        engine.isready()

        # If we starve the engine on time, it may not find a move
        tries = 1
        score, sf_move = {}, None
        while 1 not in score or sf_move is None:
            sf_move = engine.go(movetime=MOVE_TIME*tries).bestmove
            score = info_handler.info["score"]
            tries += 1
            if tries > 10:
                print('Warning: stockfish not returning moves in position', score, sf_move, file=sys.stderr)
                print(board, file=sys.stderr)

        # Don't take too many examples from each game
        if random.random() < .1:
            yield model.prepare_example(board, sf_move, score[1])

        # Find a move to make
        if selfplay_model and board.fullmove_number < 30:
            move, _ = selfplay_model.find_move(board, max_labels=10, pick_random=True)
        else:
            move = sf_move
            # In the beginning of the game, add some randomness
            if random.random()*board.fullmove_number < 1:
                move = random.choice(list(board.legal_moves))
        board.push(move)

def run_thread(thread_id, module, example_queue):
    selfplay_model = args.model and module.Model(args.model)

    engine = chess.uci.popen_engine(STOCKFISH_PATH)
    info_handler = chess.uci.InfoHandler()
    engine.info_handlers.append(info_handler)
    engine.uci()
    engine.isready()
    start, last = time.time(), 0
    for i in range(N_GAMES//THREADS):
        # Predicting progress and ETA
        if thread_id == 0 and time.time()-last > .5:
            pg = i*THREADS/N_GAMES
            if i == 0: pg += 1/1000000
            etr = (time.time() - start)*(1/pg-1)
            print('Progress: {:.1f}%. Remaining: {}'
                  .format(pg*100, str(timedelta(seconds=int(etr)))),
                  file=sys.stderr, end='\r')
            last = time.time()

        # Play a game
        for line in play_game(engine, info_handler, module, selfplay_model):
            example_queue.put(line)
    example_queue.put(None)

    if thread_id == 0:
        print()
        print('Finishing remaining threads...', file=sys.stderr)

def main():
    if args.alg == 'fasttext':
        import fastchess
        module = fastchess
    elif args.alg == 'tensor':
        import tensorsketch
        module = tensorsketch

    example_handler = module.ExampleHandler(args.to)
    queue = multiprocessing.Queue()

    ps = []
    for i in range(THREADS):
        p = multiprocessing.Process(target=run_thread, args=(i, module, queue))
        p.start()
        ps.append(p)

    remaining = len(ps)
    while remaining != 0:
        val = queue.get()
        if val is None:
            remaining -= 1
        else:
            example_handler.add(val)

    for p in ps:
        p.join()

    example_handler.done()



if __name__ == '__main__':
    main()

