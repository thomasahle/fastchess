import chess
import fastchess
from play_chess import MCTS_Model
import sys
import argparse

# TODO: Inspiration from https://github.com/mdoege/sunfish/blob/master/uci.py

# Disable buffering
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

NAME = 'FastChess'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Location of fasttext model to use')
    parser.add_argument('-occ', action='store_true', help='Add -Occ features')
    parser.add_argument('-rand', nargs='?', metavar='TEMP', const=1, default=0, type=float, help='Play random moves from the posterior distribution to the 1/temp power.')
    args = parser.parse_args()

    fastchess_model = fastchess.Model(args.model_path, occ=args.occ)
    model = MCTS_Model(fastchess_model, pvs=0, uci_format=True)
    board = chess.Board()

    our_time, opp_time = 1000, 1000 # time in centi-seconds

    print(NAME)

    stack = []
    while True:
        if stack:
            smove = stack.pop()
        else: smove = input()

        if smove == 'quit':
            break

        elif smove == 'uci':
            print('uciok')

        elif smove == 'isready':
            print('readyok')

        elif smove == 'ucinewgame':
            stack.append(f'position fen {chess.STARTING_FEN}')

        elif smove.startswith('position'):
            params = smove.split(' ', 2)
            if params[1] == 'fen':
                board = chess.Board(params[2])
            elif params[1] == 'startpos':
                board = chess.Board()
                if len(params) > 2:
                    params = params[2].split(' ')
                    if params[0] == 'moves':
                        for move in params[1:]:
                            board.push(chess.Move.from_uci(move))
            else:
                print(f'Did not understand position {params}')

        elif smove.startswith('go'):
            params = smove.split(' ')
            params = dict(zip(*[iter(params[1:])]*2))
            rolls = int(params.get('nodes', 0))
            movetime = int(params.get('movetime', 0))/1000
            wtime = int(params.get('wtime', 0))/1000
            btime = int(params.get('btime', 0))/1000
            movestogo = int(params.get('movestogo', 40))
            print(params, '...')

            if not movetime and wtime and btime:
                if board.turn == chess.WHITE:
                    movetime = wtime/(movestogo)
                else:
                    movetime = btime/(movestogo)

            print('going', movetime, wtime, btime)

            move = model.find_move(board, rolls=rolls, movetime=movetime,
                                   debug=False, temperature=args.rand)
            print('bestmove', move)

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])

        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        #elif smove.startswith('wtime'):
            #our_time = int(smove.split()[1])

        #elif smove.startswith('btime'):
            #opp_time = int(smove.split()[1])

        else:
            print(f'info string Ignoring command {smove}')

if __name__ == '__main__':
    main()
