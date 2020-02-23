import itertools
import numpy as np
import scipy.sparse as sp
import chess, chess.pgn
import random
from pathlib import Path
import argparse
import concurrent.futures
import functools
import pyspark
from pystreams.pystreams import Stream
#from pyspark.context import SparkContext
#sc = SparkContext('local[*]', 'test')
#sc = SparkContext('test')


    # TODO: Consider input features for
    # - castling and ep-rights
    # - past move
    # - occupied squares (makes it easier to play legally)
    # - whether the king is in check
    # - attackers/defenders fr each square
def binary_encode(board):
    """ Returns the board as a binary vector, for eval prediction purposes. """
    rows = []
    for color in [chess.WHITE, chess.BLACK]:
        for ptype in range(chess.PAWN, chess.KING + 1):
            mask = board.pieces_mask(ptype, color)
            rows.append(list(map(int, bin(mask)[2:].zfill(64))))
    ep = [0] * 64
    if board.ep_square:
        ep[board.ep_square] = 1
    rows.append(ep)
    rows.append([
        int(board.turn),
        int(bool(board.castling_rights & chess.BB_A1)),
        int(bool(board.castling_rights & chess.BB_H1)),
        int(bool(board.castling_rights & chess.BB_A8)),
        int(bool(board.castling_rights & chess.BB_H8)),
        int(board.is_check())
    ])
    return np.concatenate(rows)

def encode_move(move):
    if move.promotion:
        return 64**2 + move.promotion - 1
    return move.from_square + move.to_square * 64

def process(node):
    board =  binary_encode(node.parent.board())
    res = node.root().headers['Result']
    score = 0
    if res == '1-0': score = int(not node.board().turn) # turn has already been changed
    elif res == '0-1': score = int(node.board().turn)
    elif res == '1/2-1/2': score = 1/2
    move = encode_move(node.move)
    ar = np.concatenate((board, [score, move]))
    #return sp.csr_matrix(ar)
    return ar

def get_games(path, max_size=None):
    import chess.pgn
    games = iter(lambda: chess.pgn.read_game(open(path)), None)
    if max_size is None:
        yield from games
    for i, game in enumerate(games):
        if i >= max_size:
            break
        yield game

def merge(it, n=1000):
    while True:
       chunk = list(itertools.islice(it, n))
       if not chunk:
           return
       yield sp.csr_matrix(chunk)

def work_spark(args):
    conf = pyspark.SparkConf().setAppName( "temp1" ).setMaster( "local[*]" ).set( "spark.driver.host", "localhost" ) \
            .set('spark.executor.memory', '6g')
    with pyspark.SparkContext("local[*]", "PySparkWordCount", conf=conf) as sc:
        (sc.parallelize(args.files)
                .flatMap(get_games)
                .flatMap(lambda game: game.mainline())
                #.sample(False, .1)
                .map(process)
                .mapPartitions(merge)
                .saveAsPickleFile('pikle.out')
                )

def work_streams(args):
    Stream(args.files) \
            .peek(print) \
            .flatmap(get_games) \
            .peek(lambda _: print('g')) \
            .flatmap(lambda game: game.mainline()) \
            .map(process) \
            .foreach(print)
            #.sample(.1) \

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', help='glob for pgn files, e.g. **/*.pgn', nargs="+")
    parser.add_argument('-test', help='test out')
    parser.add_argument('-train', help='train out')
    parser.add_argument('-ttsplit', default=.8, help='test train split')
    parser.add_argument('-eval', action='store_true',
                        help='predict eval rather than moves')
    args = parser.parse_args()

    work_spark(args)
    #work_streams(args)
    print('Done!')

if __name__ == '__main__':
    main()

