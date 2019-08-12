import chess.pgn
import random
import scipy as sp
import scipy.sparse
import numpy as np
import os
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('files', help='glob for pgn files, e.g. **/*.pgn')
parser.add_argument('-xout', help='input')
parser.add_argument('-yout', help='labels')
parser.add_argument('-moves', action='store_true', help='predict moves')
args = parser.parse_args()

# Used for games
POS_PER_GAME = 2


def encode_move(move):
    if move.promotion:
        return 64**2 + move.promotion - 1
    return move.from_square + move.to_square*64


rows = []
ress = []
last_print = 0
for p in Path('.').glob(args.files):
    print('Doing', p)
    with open(p) as file:
        for game in iter(lambda: chess.pgn.read_game(file), None):
            if len(rows) >= last_print + 100:
                last_print = len(rows)
                print(len(rows), end='\r')

            if args.moves:
                for node in game.mainline():
                    rows.append(encode(node.board()))
                    ress.append(encode_move(node.move))
            else:
                res = {'1-0': 1, '0-1': -1,
                       '1/2-1/2': 0}[game.headers['Result']]
                ress += [res]*POS_PER_GAME

                nodes = random.sample(list(game.mainline()), POS_PER_GAME)
                positions = [encode(node.board()) for node in nodes]
                rows += random.sample(positions, POS_PER_GAME)

    print()
    print('Combining rows')
    # Todo: there is also sp.sparse.vstack
    m = np.array(rows)
    print('Making sparse')
    s = sp.sparse.csr_matrix(m)
    save_as = 'games.out'
    print(f'Saving as {args.xout}')
    sp.sparse.save_npz(args.xout, s)
    print(f'Saving results as {args.yout}')
    np.save(args.yout, np.array(ress))

print(f'Got {len(rows)} positions!')
print('Done!')
