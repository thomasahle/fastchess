import math
import hashlib
import functools
import logging
import sys
import json
import pathlib
import asyncio
import argparse
import textwrap
import warnings
import itertools
import re
import os
from collections import namedtuple

import chess.pgn
import chess.engine
import chess
import numpy as np
import skopt

from arena import Arena

warnings.filterwarnings(
    'ignore',
    message='The objective has been evaluated at this point before.')

class Formatter(argparse.HelpFormatter):

    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))

    def _get_help_string(self, action):
        help = action.help
        if not action.default:
            return help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        return help

parser = argparse.ArgumentParser(
        formatter_class=Formatter,
        fromfile_prefix_chars='@',
        usage='%(prog)s ENGINE_NAME [options]',
        description=textwrap.dedent('''
            Tune.py is a tool that allows you to tune chess engines with black
            box optimization through Scikit-Optimize (or skopt). Engine
            communication is handled through python-chess, so all you need is an
            uci or cecp compatible engine supporting options, and you are set!
            \n\n
            Simple example for tuning the MilliCpuct option of fastchess:
            $ python tune.py fastchess -opt MilliCpuct
            \n\n
            Tune.py uses an engine.json file to load engines. A simple such file
            is provided in the git repositiory and looks something like this:
            [{
              "name": "stockfish",
              "command": "stockfish",
              "protocol": "uci"
            }]
            \n\n
            Tip: If you have too many options to handle, tune.py can read its
            arguemnts from a file with `tune.py @argumentfile`.
            '''),
        )
parser.add_argument('-debug', nargs='?', metavar='PATH', const=sys.stdout,
                    default=None, type=pathlib.Path,
                    help='Enable debugging of engines.')
parser.add_argument('-log-file', metavar='PATH', type=pathlib.Path,
                    help='Used to recover from crashes')
parser.add_argument('-n', type=int, default=100,
                    help='Number of iterations')
parser.add_argument('-concurrency', type=int, default=1, metavar='N',
                    help='Number of concurrent games')
parser.add_argument('-games-file', metavar='PATH', type=pathlib.Path,
                    help='Store all games to this pgn')

group = parser.add_argument_group('Engine options')
group.add_argument('engine', metavar='ENGINE_NAME',
                   help='Engine to tune')
group.add_argument('-conf', type=pathlib.Path, metavar='PATH',
                   help='Engines.json file to load from')
group.add_argument('-opp-engine', metavar='ENGINE_NAME',
                   help='Tune against a different engine')

group = parser.add_argument_group('Games format')
group.add_argument('-book', type=pathlib.Path, metavar='PATH',
                   help='pgn file with opening lines.')
group.add_argument('-n-book', type=int, default=10, metavar='N',
                   help='Length of opening lines to use in plies.')
group.add_argument('-games-per-encounter', type=int, default=2, metavar='N',
                   help='Number of book positions to play at each set of argument explored.')
group.add_argument('-max-len', type=int, default=10000, metavar='N',
                   help='Maximum length of game in plies before termination.')
subgroup = group.add_mutually_exclusive_group(required=True)
subgroup.add_argument('-movetime', type=int, metavar='MS',
                      help='Time per move in ms')
subgroup.add_argument('-nodes', type=int, metavar='N',
                      help='Nodes per move')

group = parser.add_argument_group('Options to tune')
group.add_argument('-opt', nargs='+', action='append', default=[],
                   metavar=('NAME', 'LOWER, UPPER'),
                   help='Integer option to tune.')
group.add_argument('-c-opt', nargs='+', action='append', default=[],
                   metavar=('NAME', 'VALUE'),
                   help='Categorical option to tune')

group = parser.add_argument_group('Optimization parameters')
group.add_argument('-base-estimator', default='GP', metavar='EST',
                   help='One of "GP", "RF", "ET", "GBRT"')
group.add_argument('-n-initial-points', type=int, default=10, metavar='N',
                   help='Number of points chosen before approximating with base estimator.')
group.add_argument('-acq-func', default='gp_hedge', metavar='FUNC',
                   help='Can be either of "LCB" for lower confidence bound.'
                   ' "EI" for negative expected improvement.'
                   ' "PI" for negative probability of improvement.'
                   ' "gp_hedge" Probabilistically chooses one of the above'
                   ' three acquisition functions at every iteration.')
group.add_argument('-acq-optimizer', default='auto', metavar='OPT',
                   help='Either "sampling" or "lbfgs"')
group.add_argument('-acq-noise', default='gaussian', metavar='VAR',
                   help='For the Gaussian Process optimizer, use this to specify the'
                   ' variance of the assumed noise. Larger values mean more exploration.')
group.add_argument('-acq-xi', default=0.01, metavar='XI', type=float,
                   help='Controls how much improvement one wants over the previous best'
                   ' values. Used when the acquisition is either "EI" or "PI".')
group.add_argument('-acq-kappa', default=1.96, metavar='KAPPA', type=float,
                   help='Controls how much of the variance in the predicted values should be'
                   ' taken into account. If set to be very high, then we are favouring'
                   ' exploration over exploitation and vice versa. Used when the acquisition'
                   ' is "LCB".')

group = parser.add_argument_group('Adjudication options')
group.add_argument('-win-adj', nargs='*', metavar='ADJ',
                   help='Adjudicate won game. Usage: '
                   '-win-adj count=4 score=400 '
                   'If the last 4 successive moves of white had a score of '
                   '400 cp or more and the last 4 successive moves of black '
                   'had a score of -400 or less then that game will be '
                   'adjudicated to a win for white. When the situation is '
                   'reversed black would win. '
                   f'Default values: count=4, score={Arena.MATE_SCORE}')


async def load_engine(engine_args, name, debug=False):
    assert engine_args and any(a['name'] == name for a in engine_args), \
            f'Engine "{name}" was not found in engines.json file'
    args = next(a for a in engine_args if a['name'] == name)
    curdir = str(pathlib.Path(__file__).parent.parent)
    popen_args = {'env': {'PATH': os.environ['PATH']}}
    if 'workingDirectory' in args:
        popen_args['cwd'] = args['workingDirectory'].replace('$FILE', curdir)
    cmd = args['command'].split()
    if cmd[0] == '$PYTHON':
        cmd[0] = sys.executable
    if args['protocol'] == 'uci':
        _, engine = await chess.engine.popen_uci(cmd, **popen_args)
    elif args['protocol'] == 'xboard':
        _, engine = await chess.engine.popen_xboard(cmd, **popen_args)
    if hasattr(engine, 'debug'):
        engine.debug(debug)
    return engine


def load_conf(conf):
    if not conf:
        path = pathlib.Path(__file__).parent.parent / 'engines.json'
        assert path.is_file(), 'No engines conf specified and unable to locate' \
                               ' engines.json file automatically.'
        return json.load(path.open())
    else:
        assert conf.is_file(), f'Unable to open "{conf}"'
        return json.load(conf.open())


def plot_optimizer(opt, lower, upper):
    import matplotlib.pyplot as plt
    plt.set_cmap("viridis")

    if not opt.models:
        print('Can not plot opt, since models do not exist yet.')
        return
    model = opt.models[-1]
    x = np.linspace(lower, upper).reshape(-1, 1)
    x_model = opt.space.transform(x)

    # Plot Model(x) + contours
    y_pred, sigma = model.predict(x_model, return_std=True)
    plt.plot(x, -y_pred, "g--", label=r"$\mu(x)$")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([-y_pred - 1.9600 * sigma,
                             (-y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.2, fc="g", ec="None")

    # Plot sampled points
    plt.plot(opt.Xi, -np.array(opt.yi),
             "r.", markersize=8, label="Observations")

    # Adjust plot layout
    plt.grid()
    plt.legend(loc='best')
    plt.show()


def x_to_args(x, dim_names, options):
    args = {name: val for name, val in zip(dim_names, x)}
    for name, val in args.items():
        opt = options[name]
        if opt.type == 'combo':
            args[name] = opt.var[val]
    return args


class DataLogger:
    def __init__(self, path, key):
        self.path = path
        self.key = key
        self.append_file = None

    def load(self, opt):
        if not self.path.is_file():
            print(f'Unable to open {self.path}')
            return 0
        print(f'Reading {self.path}')
        xs, ys = [], []
        with self.path.open('r') as file:
            for line in file:
                obj = json.loads(line)
                if obj.get('args') == self.key:
                    x, y = obj['x'], obj['y']
                    print(f'Using {x} => {y} from log-file')
                    try:
                        xs.append(x)
                        ys.append(-y)
                    except ValueError as e:
                        print('Ignoring bad data point', e)
        # Fit the first model, but because of a bug the lists can't be empty.
        if xs:
            print('Fitting first model')
            opt.tell(xs, ys, fit=True)
        return len(xs)

    def store(self, x, y):
        if not self.append_file:
            self.append_file = self.path.open('a')
        x = [xi if type(xi) == str else float(xi) for xi in x]
        y = float(y)
        print(json.dumps({'args': self.key, 'x': x, 'y': y}),
              file=self.append_file, flush=True)


def load_book(path, n_book):
    if not path.is_file():
        print(f'Error: Can\'t open book {path}.')
        return
    with open(path, encoding='latin-1') as file:
        for game in iter((lambda: chess.pgn.read_game(file)), None):
            board = game.board()
            for _, move in zip(range(n_book), game.mainline_moves()):
                board.push(move)
            yield board


def parse_options(opts, copts, engine_options):
    dim_names = []
    dimensions = []
    for name, *lower_upper in opts:
        opt = engine_options.get(name)
        if not opt:
            if not lower_upper:
                print(f'Error: engine has no option {name}. For hidden options'
                      ' you must specify lower and upper bounds.')
                continue
            else:
                print(f'Warning: engine has no option {name}')
        dim_names.append(name)
        lower, upper = map(int, lower_upper) if lower_upper else (opt.min, opt.max)
        dimensions.append(skopt.utils.Integer(lower, upper, name=name))
    for name, *categories in copts:
        opt = engine_options.get(name)
        if not opt:
            if not categories:
                print(f'Error: engine has no option {name}. For hidden options'
                      ' you must manually specify possible values.')
                continue
            else:
                print(f'Warning: engine has no option {name}')
        dim_names.append(name)
        cats = categories or opt.var
        cats = [opt.var.index(cat) for cat in cats]
        dimensions.append(skopt.utils.Categorical(cats, name=name))
    if not dimensions:
        print('Warning: No options specified for tuning.')
    return dim_names, dimensions


def summarize(opt, samples):
    X = np.vstack((opt.space.rvs(n_samples=samples), opt.Xi))
    Xt = opt.space.transform(X)
    y_pred, sigma = opt.models[-1].predict(Xt, return_std=True)
    y_pred = -y_pred  # Change to maximization
    for kappa in range(4):
        i = np.argmax(y_pred - kappa * sigma)

        def score_to_elo(score):
            if score <= -1:
                return float('inf')
            if score >= 1:
                return -float('inf')
            return - 400 * math.log10(2 / (score + 1) - 1)
        elo = score_to_elo(y_pred[i] / 2)
        pm = max(abs(score_to_elo(y_pred[i] + sigma[i]) - elo),
                 abs(score_to_elo(y_pred[i] + sigma[i]) - elo))
        print(f'Best expectation (κ={kappa}): {X[i]}'
              f' = {y_pred[i]/2:.3f} ± {sigma[i]/2:.3f}'
              f' (ELO-diff {elo:.3f} ± {pm:.3f})')


async def main():
    args = parser.parse_args()

    if args.debug:
        if args.debug == sys.stdout:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=args.debug, filemode='w')

    # Do not run the tuner if something is wrong with the adjudication option
    # that is set by the user. These options could be critical in tuning.
    win_adj_count, win_adj_score = 4, Arena.MATE_SCORE
    if args.win_adj:
        for n in args.win_adj:
            m = re.match('count=(\d+)', n)
            if m:
                win_adj_count = int(m.group(1))
            m = re.match('score=(\d+)', n)
            if m:
                win_adj_score = int(m.group(1))

    book = []
    if args.book:
        book.extend(load_book(args.book, args.n_book))
        print(f'Loaded book with {len(book)} positions')
    if not book:
        book.append(chess.Board())
        print('No book. Starting every game from initial position.')

    print('Loading engines')
    conf = load_conf(args.conf)
    engines = await asyncio.gather(*(asyncio.gather(
        load_engine(conf, args.engine),
        load_engine(conf, args.opp_engine or args.engine))
        for _ in range(args.concurrency)))
    options = engines[0][0].options

    print('Parsing options')
    dim_names, dimensions = parse_options(args.opt, args.c_opt, options)

    opt = skopt.Optimizer(
        dimensions,
        base_estimator=args.base_estimator,
        n_initial_points=args.n_initial_points,
        acq_func=args.acq_func,
        acq_optimizer=args.acq_optimizer,
        acq_func_kwargs={
            'xi': args.acq_xi,
            'kappa': args.acq_kappa,
            'noise': args.acq_noise
            }
    )

    if args.games_file:
        games_file = args.games_file.open('a')
    else:
        games_file = sys.stdout

    if args.log_file:
        key_args = args.__dict__.copy()
        # Not all arguments change the result, so no need to keep them in the key.
        for key in ('n', 'games_file', 'concurrency'):
            del key_args[key]
        key = repr(sorted(key_args.items())).encode()
        data_logger = DataLogger(args.log_file, key=hashlib.sha256(key).hexdigest())
        cached_games = data_logger.load(opt)
    else:
        data_logger = None
        cached_games = 0

    limit = chess.engine.Limit(
        nodes=args.nodes,
        time=args.movetime and args.movetime / 1000)

    assert args.games_per_encounter >= 2 and args.games_per_encounter % 2 == 0, \
            'Games per encounter must be even and >= 2.'

    # Run tasks concurrently
    try:
        started = cached_games

        def on_done(task):
            if task.exception():
                logging.error('Error while excecuting game')
                task.print_stack()

        def new_game(arena):
            x = opt.ask()
            engine_args = x_to_args(x, dim_names, options)
            print(f'Starting {args.games_per_encounter} games {started}/{args.n} with {engine_args}')
            async def routine():
                await arena.configure(engine_args)
                return await arena.run_games(book, game_id=started,
                                             games_played=args.games_per_encounter)
            task = asyncio.create_task(routine())
            # We tag the task with some attributes that we need when it finishes.
            setattr(task, 'tune_x', x)
            setattr(task, 'tune_arena', arena)
            setattr(task, 'tune_game_id', started)
            task.add_done_callback(on_done)
            return task
        tasks = []
        xs = opt.ask(min(args.concurrency, args.n - started)
                     ) if args.n - started > 0 else []
        for conc_id, x_init in enumerate(xs):
            enginea, engineb = engines[conc_id]
            arena = Arena(
                enginea,
                engineb,
                limit,
                args.max_len,
                win_adj_count,
                win_adj_score)
            tasks.append(new_game(arena))
            started += 1
        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = list(pending)
            for task in done:
                res = task.result()
                arena, x, game_id = task.tune_arena, task.tune_x, task.tune_game_id
                games, y, er = res
                y /= args.games_per_encounter # Normalize to [-1, 1] range
                for game in games:
                    print(game, end='\n\n', file=games_file, flush=True)
                if er:
                    print('Game erred:', er, type(er))
                    continue
                opt.tell(x, -y)  # opt is minimizing
                # Delete old models to save memory. Note that for the first 10 tells (default)
                # or the specified -n-initial-points no model is created, as we are still just
                # querying at random.
                logging.debug(f'Number of models {len(opt.models)}')
                if len(opt.models) > 1:
                    del opt.models[0]
                results = ', '.join(g.headers['Result'] for g in games)
                print(f'Finished game {game_id} {x} => {y} ({results})')
                if data_logger:
                    data_logger.store(x, y)
                if started < args.n:
                    tasks.append(new_game(arena))
                    started += 1
    except asyncio.CancelledError:
        pass

    if opt.Xi and opt.models:
        print('Summarizing best values')
        summarize(opt, samples=args.n)
        if len(dimensions) == 1:
            plot_optimizer(opt, dimensions[0].low, dimensions[0].high)
    else:
        print('Not enought data to summarize results.')

    logging.debug('Quitting engines')
    try:
        # Could also use wait here, but wait for some reason fails if the list
        # is empty. Why can't we just wait for nothing?
        await asyncio.gather(*(e.quit() for es in engines for e in es
                               if not e.returncode.done()))
    except chess.engine.EngineError:
        pass


if __name__ == '__main__':
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    try:
        if hasattr(asyncio, 'run'):
            asyncio.run(main())
        else:
            asyncio.get_event_loop().run_until_complete(main())
    except KeyboardInterrupt:
        logging.debug('KeyboardInterrupt at root')
