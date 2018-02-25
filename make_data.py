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
import fastText

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
parser.add_argument('-selfplay', type=str, default=None,
                    help='Fastchess model to play with')
args = parser.parse_args()

MOVE_TIME = args.movetime
THREADS = args.threads
N_GAMES = args.games

def play_game(engine, info_handler, selfplay_model=None):
    board = chess.Board()
    engine.ucinewgame()
    while not board.is_game_over():
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

        # Always predict from Make fasttext line
        if board.turn == chess.WHITE:
            string = ' '.join(fastchess.board_to_words(board))
            uci_move = sf_move.uci()
        else:
            string = ' '.join(fastchess.board_to_words(board.mirror()))
            uci_move = fastchess.mirror_move(sf_move).uci()

        labels = []
        labels.append('__label__' + str(fastchess.discretize_score(score[1])))
        labels.append('__label__' + uci_move)
        labels.append('__label__f_' + uci_move[:2])
        labels.append('__label__t_' + uci_move[2:])
        yield string + ' ' + ' '.join(labels)

        # Find a move to make
        if selfplay_model and board.fullmove_number < 30:
            move, _ = fastchess.find_move(selfplay_model, board,
                                          max_labels=10, pick_random=True)
        else:
            move = sf_move
            # In the beginning of the game, add some randomness
            if 1/board.fullmove_number > random.random():
                move = random.choice(list(board.legal_moves))
        board.push(move)

def run_thread(thread_id, print_lock):
    selfplay_model = args.selfplay and fastText.load_model(args.selfplay)
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
                  file=sys.stderr)
            last = time.time()

        # Play a game
        for line in play_game(engine, info_handler, selfplay_model):
            with print_lock:
                print(line)

    if thread_id == 0:
        print('Finishing remaining threads...', file=sys.stderr)

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

