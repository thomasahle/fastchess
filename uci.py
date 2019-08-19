import chess
import sys
import re
import itertools
from collections import namedtuple
from enum import Enum
import time
import os.path

import fastchess
import mcts
from controller import MCTS_Controller


# Disable buffering.
# For some reason this is often not enough, so you might have to run uci.py
# as `python -u uci.py` to really get unbuffered input/output.

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        sys.stderr.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)
sys.stderr = open('log', 'a')


# There is also chess.engine.Option
Type_Spin = namedtuple('spin', ['default', 'min', 'max'])
Type_String = namedtuple('string', ['default'])
Type_Check = namedtuple('check', ['default'])
Type_Button = namedtuple('button', [])
Type_Combo = namedtuple('combo', ['default', 'vars'])

class UCI:
    def __init__(self):
        self.debug = False
        self.board = chess.Board()
        self.option_types = {
            'ModelPath': Type_String(default='models/model.bin'),
            # Play random moves from the posterior distribution to the temp/100 power.
            'Temperature': Type_Spin(default=0, min=0, max=100),
            'MultiPV': Type_Spin(default=0, min=0, max=10),
            'MilliCPUCT': Type_Spin(default=2000, min=0, max=1000000),
            'Cache': Type_Check(default=True),
            'PolicySoftmaxTemp': Type_Spin(default=100, min=1, max=10000),
            'UFormula': Type_Spin(default=0, min=0, max=4),

            # Legal moves will get at least this policy score
            'LegalPolicyTreshold': Type_Spin(default=1, min=-100, max=100),
            'CapturePolicyTreshold': Type_Spin(default=2, min=-100, max=100),
            'CheckPolicyTreshold': Type_Spin(default=2, min=-100, max=100),

            # TODO: Leela has many interesting options that we may consider, like
            # fpu-value (We use -.99), fpu-at-root (they use 1), policy-softmax-temp
            # (somebody said 2.2 is a good value)
            # See https://github.com/LeelaChessZero/lc0/wiki/Lc0-options for more

            # Migrated from play_chess
            # parser.add_argument('-rand', nargs='?', help='Play random moves from the posterior distribution to the 1/temp power.',
            # metavar='TEMP', const=1, default=0, type=float)
            # parser.add_argument('-profile', action='store_true', help='Run profiling. (Only with selfplay)')
        }
        self.options = {key: val.default for key, val in self.option_types.items()
                        if hasattr(val, 'default')}

        # These objects depend on options to be set. (ModelPath in particular.)
        self.fastchess_model = None
        self.controller = None
        self.setoption(None, None)

        # Statistics for time management
        self.nps = 0
        self.roll_kldiv = 1
        self.tot_rolls = 0

    def parse(self, line):
        cmd, arg = line.split(' ', 1) if ' ' in line else (line, '')
        if cmd == 'uci':
            self.uci()
        elif cmd == 'debug':
            self.set_debug(arg != 'off')
        elif cmd == 'isready':
            self.isready()
        elif cmd == 'setoption':
            i = arg.find('value')
            if i >= 0:
                name = arg[len('name '):i - 1]
                value = arg[i + len('value '):]
            else:
                name = arg[len('name '):]
            opt = self.option_types.get(name)
            if not opt:
                print(f'Did not understand option "{cmd}"', file=sys.stderr)
            elif type(opt) == Type_Spin:
                value = int(value)
            elif type(opt) == Type_Check:
                value = (value == 'true')
            self.setoption(name, value)
        elif cmd == 'ucinewgame':
            self.ucinewgame()
        elif cmd == 'position':
            args = arg.split()
            if args[0] == 'fen':
                board = chess.Board(' '.join(args[1:7]))
                moves = args[8:]
            else:
                assert args[0] == 'startpos'
                board = chess.Board()
                moves = args[2:]
            self.position(board, map(chess.Move.from_uci, moves))
        elif cmd == 'go':
            args = arg.split()
            params = {}
            while args:
                key, *args = args
                if key == 'searchmoves':
                    def uci_or_none(string):
                        try:
                            return chess.Move.from_uci(string)
                        except ValueError:
                            return None
                    params['searchmoves'] = list(itertools.takewhile(
                        (lambda x: x), map(uci_or_none, args[1:])))
                    del args[:len(params['searchmoves'])]
                elif key in ('ponder', 'infinite'):
                    params[key] = True
                else:
                    params[key] = int(args[0])
                    del args[0]
            self.go(**params)
        elif cmd == 'stop':
            self.stop()
        elif cmd == 'ponderhit':
            self.ponderhit()
        else:
            print(f'info string Ignoring command {cmd}')

    def uci(self):
        print(f'id name FastChess')
        print(f'id author Thomas Dybdahl Ahle')
        for name, op in self.option_types.items():
            parts = [f'option name {name} type {type(op).__name__}']
            if type(op) == Type_Spin:
                parts.append(f'default {op.default} min {op.min} max {op.max}')
            if type(op) == Type_Check:
                parts.append(f'default {op.default}')
            if type(op) == Type_Combo:
                parts.append(f'default {op.default}')
                for var in op.vars:
                    parts.append(f'var {var}')
            if type(op) == Type_Button:
                pass
            if type(op) == Type_String:
                parts.append(f'default {op.default}')
            print(' '.join(parts))
        print('uciok')

    def set_debug(self, debug=True):
        self.debug = debug

    def isready(self):
        print('readyok')

    def ucinewgame(self):
        pass

    def setoption(self, name, value=None):
        self.options[name] = value

        model_path = self.options.get('ModelPath')
        if model_path and self.fastchess_model is None:
            if not os.path.isfile(model_path):
                print(f'error path {model_path} not found.')
            else:
                self.fastchess_model = fastchess.Model(self.options['ModelPath'])

        self.controller = MCTS_Controller(args=mcts.Args(
            model=self.fastchess_model,
            debug=self.debug,
            cpuct=self.options['MilliCPUCT'] / 1000,
            legal_t=self.options['LegalPolicyTreshold'],
            cap_t=self.options['CapturePolicyTreshold'],
            chk_t=self.options['CheckPolicyTreshold'],
            ))

    def position(self, board, moves):
        self.board = board
        for move in moves:
            self.board.push(move)

    def go(self, searchmoves=(), ponder=False, wtime=0, btime=0, winc=0, binc=0,
           movestogo=40, depth=0, nodes=0, mate=0, movetime=0, infinite=False):
        """ See UCI documentation for what the options mean. """
        if searchmoves or ponder or mate or depth:
            print('info string Ignoring unsupported go options')

        min_kldiv = max_time = max_rolls = 0

        if movetime:
            max_time = movetime / 1000

        elif nodes:
            max_rolls = nodes

        else:
            time_left = wtime / 1000 if self.board.turn == chess.WHITE else btime / 1000
            inc = winc / 1000 if self.board.turn == chess.WHITE else binc / 1000
            time_per_move = time_left / (movestogo + 1) + inc

            # If the opponent has a lot less time than us, we might as well use a bit more.
            # This kinda assumes we played with the same timecontrol from the beginning.
            opp_time_per_move = (wtime + btime) / 1000 / \
                (movestogo + 1) + (winc + binc) / 1000 - time_per_move
            if opp_time_per_move:
                time_ratio = (time_per_move + 1) / (opp_time_per_move + 1)
                print(f'info string time ratio {time_ratio:.3}')
                time_per_move *= time_ratio**.5
                print(f'info string time per move {time_per_move:.1f}')

            # Try to convert time_per_move into a number of rolls
            if self.tot_rolls == 0:
                max_time = time_per_move
                min_kldiv = 0
            else:
                # We allow a fair bit of extra time to try and allow the kl_div
                # mechanism to really work.
                max_time = min(2 * time_per_move, time_left / 2)
                mean_rolls = self.nps * time_per_move
                min_kldiv = 1 / mean_rolls

        # See that some kind of condition has been set
        if not infinite and not (max_time or min_kldiv or max_rolls):
            print('info string No time, using priors directly.')
            use_mcts = False
        else:
            use_mcts = True

        temp = self.options['Temperature'] / 100
        node, stats = self.controller.find_move(self.board, min_kldiv=min_kldiv, max_rolls=max_rolls,
                                                max_time=max_time, temperature=temp, pvs=self.options['MultiPV'], use_mcts=use_mcts)

        if use_mcts:
            # Conservative discounting using the harmonic mean
            if self.nps:
                self.nps = 1 / (.5 / self.nps + .5 / (stats.rolls / stats.elapsed))
            else:
                self.nps = stats.rolls / stats.elapsed

            # Don't use discounting when guessing the constant C such that kl_div =
            # C/rolls.
            self.roll_kldiv = (stats.kl_div * stats.rolls**2 + self.roll_kldiv *
                               self.tot_rolls) / (self.tot_rolls + stats.rolls)
            self.tot_rolls += stats.rolls
            print(
                f'info string roll_kldiv {self.roll_kldiv:.1f} rolls {stats.rolls} kl_div {stats.kl_div/1:.1} tot {self.tot_rolls}')

        # Hack to ensure we can always get the bestmove from python-chess
        print(f'info pv {node.move.uci()}')

        parts = ['bestmove', node.move.uci()]

        if node.children:
            ponder_node = max(node.children, key=lambda n: n.N)
            parts += ['ponder', ponder_node.move.uci()]

        print(' '.join(parts))

    def stop(self):
        self.controller.stop()

    def ponderhit(self):
        pass


def main():
    uci = UCI()
    while True:
        cmd = input()
        print('stderr', cmd, file=sys.stderr)
        if cmd == 'quit':
            break
        uci.parse(cmd)


if __name__ == '__main__':
    main()
