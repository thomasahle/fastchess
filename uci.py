import chess
import sys
import re
import itertools
from collections import namedtuple
import os.path
from enum import Enum
import random
import select

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
#print('Moving stderr to log', file=sys.stderr)
sys.stderr = open('log', 'a')


class CaseInsensitiveDict(dict):
    def __init__(self, *args, **kwargs):
        super(CaseInsensitiveDict, self).__init__(*args, **kwargs)
        self._convert_keys()

    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(key.lower())

    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(key.lower(), value)

    def __delitem__(self, key):
        return super(CaseInsensitiveDict, self).__delitem__(key.lower())

    def __contains__(self, key):
        return super(CaseInsensitiveDict, self).__contains__(key.lower())

    def has_key(self, key):
        return super(CaseInsensitiveDict, self).has_key(key.lower())

    def pop(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).pop(key.lower(), *args, **kwargs)

    def get(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).get(key.lower(), *args, **kwargs)

    def setdefault(self, key, *args, **kwargs):
        return super(CaseInsensitiveDict, self).setdefault(key.lower(), *args, **kwargs)

    def update(self, E={}, **F):
        super(CaseInsensitiveDict, self).update(self.__class__(E))
        super(CaseInsensitiveDict, self).update(self.__class__(**F))

    def _convert_keys(self):
        for k in list(self.keys()):
            v = super(CaseInsensitiveDict, self).pop(k)
            self.__setitem__(k, v)


# There is also chess.engine.Option
Type_Spin = namedtuple('spin', ['default', 'min', 'max'])
Type_String = namedtuple('string', ['default'])
Type_Check = namedtuple('check', ['default'])
Type_Button = namedtuple('button', [])
Type_Combo = namedtuple('combo', ['default', 'vars'])


class State(Enum):
    Searching = 1
    Pondering = 2
    Transitioning = 3
    Inactive = 4
    Stopping = 5


class UCI:
    def __init__(self):
        self.debug = False
        self.state = State.Inactive
        self.board = chess.Board()
        self.option_types = {
            'ModelPath': Type_String(default='models/model.bin'),
            # Play random moves from the posterior distribution to the temp/100 power.
            'Temperature': Type_Spin(default=0, min=0, max=100),
            'MultiPV': Type_Spin(default=0, min=0, max=10),
            # Cpuct 4.3 seems good with current treshold values. Previously 2.8 seemed
            # better.
            'MilliCPUCT': Type_Spin(default=4300, min=0, max=10000),

            # Legal moves will get at least this policy score/100
            'LegalPolicyTreshold': Type_Spin(default=-3568, min=-10000, max=10000),
            'CapturePolicyTreshold': Type_Spin(default=-1639, min=-10000, max=10000),
            # Adding a bonus to moves that give check is currently not used, since
            # tuning found it irrelevant with current models. It might still be
            # usedful for sovling chess puzzles etc.
            'CheckPolicyTreshold': Type_Spin(default=-10000, min=-10000, max=10000),

            # TODO: Leela has many interesting options that we may consider, like
            # fpu-value (We use -.99), fpu-at-root (they use 1), policy-softmax-temp
            # (somebody said 2.2 is a good value)
            # See https://github.com/LeelaChessZero/lc0/wiki/Lc0-options for more
        }
        self.options = CaseInsensitiveDict({key: val.default for key, val
                                            in self.option_types.items() if hasattr(val, 'default')})

        # These objects depend on options to be set. (ModelPath in particular.)
        self.fastchess_model = None
        self.controller = None
        self.searcher = None # The iter we get from the controller
        self.setoption('', '')

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
            # UCI options are case insensitive
            opt = CaseInsensitiveDict(self.option_types).get(name)
            if not opt:
                print(f'Did not recognize option "{name}"', file=sys.stderr)
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

        self.controller_args = mcts.Args(
            model=self.fastchess_model,
            debug=self.debug,
            cpuct=self.options['MilliCPUCT'] / 1000,
            legal_t=self.options['LegalPolicyTreshold'] / 100,
            cap_t=self.options['CapturePolicyTreshold'] / 100,
            chk_t=self.options['CheckPolicyTreshold'] / 100
        )


    def beat(self):
        if self.state in [State.Searching, State.Transitioning, State.Stopping]:
            res = next(self.searcher)
            # If we stopped, it doesn't matter if we wanted to or not.
            # Maybe the time just ran out, or the game ended.
            if res is not None:
                node, stats = res
                self._finish_stopping(node, stats)
        return self.state is not State.Inactive

    def position(self, board, moves):
        assert self.state == State.Inactive
        self.board = board
        for move in moves:
            self.board.push(move)

    def go(self, **args):
        """ See UCI documentation for what the options mean. """
        assert self.state == State.Inactive
        self._start_going(**args)

    def stop(self):
        if self.state is State.Inactive:
            print(f"info Ignoring stop command while engine isn't searching. ({self.state})")
            return
        # We will get a stop if the user didn't play the pondermove.
        # We give the controller a chance to stop itself.
        self.controller.stop()
        self.state = State.Stopping

    def ponderhit(self):
        # When the opponent premoves, we don't get asked to ponder, but are
        # started in normal search directly. However LiChess still sends 'ponderhit'.
        if self.state is not State.Pondering:
            print(f"info Ignoring ponderhit command while engine isn't pondering. ({self.state})")
            return
        # We now have to transition from pondering to searching.
        # We do that by stopping the search and starting a new one in normal note.
        # Note: There may be an issue with regards to time controls by doing it this way.
        # Just doing a new search during ponder might screw up the rolling
        # kl_div computations, since they don't know that the node has been pre-
        # considered and thus assume we magically do much better than we actually do...
        self.controller.stop()
        self.state = State.Transitioning

    def _start_going(self, searchmoves=(), ponder=False, wtime=0, btime=0, winc=0, binc=0,
           movestogo=40, depth=0, nodes=0, mate=0, movetime=0, infinite=False):

        if searchmoves or mate or depth:
            print('info string Ignoring unsupported go options')

        if ponder:
            # If pondering, the search arguments aren't really for us to use now,
            # but to use once the pondering is over. We thus store them away safely
            # for use when transitioning.
            self.ponder_search_args = dict(wtime=wtime, btime=btime, winc=winc, binc=binc,
                movestogo=movestogo, depth=depth, nodes=nodes, mate=mate, movetime=movetime,
                infinite=infinite)
            infinite = True
            # Like Leela, we cheat UCI and ponder over all opponent moves
            board = self.board.copy()
            _ponder_move = board.pop()
        else:
            board = self.board

        temp = self.options['Temperature'] / 100

        min_kldiv = max_time = max_rolls = 0

        if infinite:
            max_time = 10**6

        # Make it a bit more fun to play against online
        elif board.ply() <= 3:
            max_time = 1
            temp = .5

        elif movetime:
            max_time = movetime / 1000

        elif nodes:
            max_rolls = nodes

        else:
            time_left = wtime / 1000 if board.turn == chess.WHITE else btime / 1000
            inc = winc / 1000 if board.turn == chess.WHITE else binc / 1000
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
                # mechanism to really work, but never more than half of our remaining
                # time.
                max_time = time_left / (movestogo/4 + 1) + inc
                mean_rolls = self.nps * time_per_move
                min_kldiv = self.roll_kldiv / mean_rolls

        # See that some kind of condition has been set
        if not infinite and not (max_time or min_kldiv or max_rolls):
            print('info string No time, using priors directly.')
            use_mcts = False
        else:
            use_mcts = True

        print(f'info string Searching with {min_kldiv=}, {max_rolls=}, {max_time=}, {temp=}, {use_mcts=}')
        self.controller = MCTS_Controller(self.controller_args)
        self.searcher = self.controller.find_move(
            board, min_kldiv=min_kldiv, max_rolls=max_rolls, max_time=max_time,
            temperature=temp, pvs=self.options['MultiPV'], use_mcts=use_mcts)

        self.state = State.Pondering if ponder else State.Searching

        # Just an easy (hacky?) way to remember args in _finish_searching
        self.search_args = locals()

    def _finish_stopping(self, node, stats):
        # go and finish were originally the same method.
        # This is just a hacky way to keep the variable space in sync.
        use_mcts = self.search_args['use_mcts']
        is_ponder = self.search_args['ponder']

        if use_mcts:
            # Conservative discounting using the harmonic mean
            if self.nps:
                self.nps = 1 / (.5 / self.nps + .5 / (stats.rolls / stats.elapsed))
            else:
                self.nps = stats.rolls / stats.elapsed

            # Don't use discounting when guessing the constant C such that  `kl_div = C/rolls`.
            # TODO: Is a linear relationship reasonable? Why not just translate to
            # time directly?
            roll_kldiv = stats.kl_div * stats.rolls
            self.roll_kldiv = (roll_kldiv * stats.rolls + self.roll_kldiv * self.tot_rolls) \
                / (stats.rolls + self.tot_rolls)
            self.tot_rolls += stats.rolls
            print(f'info string roll_kldiv {self.roll_kldiv:.1f} rolls {stats.rolls}'
                  f' kl_div {stats.kl_div/1:.1} tot {self.tot_rolls}')

        if is_ponder and self.state is not State.Transitioning:
            # We have to say something to indicate we stopped, but since we
            # are thinking from the 'wrong' position, it would be confusing to
            # give an actual move.
            random_move = random.choice(list(self.board.legal_moves))
            print('bestmove', random_move)

        if not is_ponder:
            # Hack to ensure we can always get the bestmove from python-chess.
            # It only gets it from pv rather than the actual bestmove response.
            print(f'info pv {node.move.uci()}')

            parts = ['bestmove', node.move.uci()]
            if node.children:
                ponder_move = max(node.children, key=lambda n: n.N).move
                parts += ['ponder', ponder_move.uci()]
            print(*parts)

        if self.state is State.Transitioning:
            self._start_going(**self.ponder_search_args)
        else:
            self.state = State.Inactive

def main():
    uci = UCI()
    while True:
        try:
            is_active = uci.beat()
            # Test if input is available on stdin.
            # Block only if the engine isn't doing anything right now.
            if not is_active or select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                #print('Taking input')
                cmd = sys.stdin.readline().strip()
                print('\n>>>', cmd, file=sys.stderr)
                if cmd == 'quit':
                    break
                uci.parse(cmd)
        except KeyboardInterrupt:
            break
    print('Good bye')

if __name__ == '__main__':
    main()

