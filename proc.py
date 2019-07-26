import chess.pgn
import random
from pathlib import Path
import argparse
import fastchess

parser = argparse.ArgumentParser()
parser.add_argument('files', help='glob for pgn files, e.g. **/*.pgn')
parser.add_argument('-test', help='test out')
parser.add_argument('-train', help='train out')
parser.add_argument('-ttsplit', default=.8, help='test train split')
args = parser.parse_args()

# TODO: Consider input features for
# - castling and ep-rights
# - past move
# - occupied squares (makes it easier to play legally)
# - whether the king is in check
# - attackers/defenders for each square

progress = 0
last_print = 0
with open(args.test, 'w') as testfile, open(args.train, 'w') as trainfile:
    for p in Path('.').glob(args.files):
        print('Doing', p)
        with open(p) as file:
            for game in iter(lambda:chess.pgn.read_game(file), None):
                if progress >= last_print + 100:
                    last_print = progress
                    print(progress, end='\r')
                    testfile.flush()
                    trainfile.flush()

                for node in game.mainline():
                    # Weirdly, the board associated with the node is the result
                    # of the move, rather than from before the move
                    line = fastchess.prepare_example(
                            node.parent.board(), node.move, 0, only_move=True)
                    print(line, file=(
                        trainfile if random.random() < args.ttsplit else testfile))
                    progress += 1

print('Done!')
