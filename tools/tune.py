import math
import hashlib
import functools
import logging
import random
import sys
import json
import pathlib
import asyncio
import argparse
import warnings

import chess.pgn
import chess.engine
import chess
import numpy as np
import skopt

warnings.filterwarnings(
    'ignore',
    message='The objective has been evaluated at this point before.')

parser = argparse.ArgumentParser()
parser.add_argument('-debug', action='store_true', help='Enable debugging of engine')
parser.add_argument('-log-file', type=pathlib.Path, help='Used to recover from crashes')
parser.add_argument('-n', type=int, default=100, help='Number of iterations')
parser.add_argument(
    '-concurrency',
    type=int,
    default=1,
    help='Number of concurrent games')
parser.add_argument('-games-file', type=pathlib.Path, help='Store all games to this pgn')

group = parser.add_argument_group('Engine options')
group.add_argument('engine', help='Engine to tune')
group.add_argument('-conf', help='Engines.json file to load from')
group.add_argument('-opp-engine', help='Tune against a different engine')

group = parser.add_argument_group('Games format')
group.add_argument('-book', type=pathlib.Path, help='pgn file with opening lines.')
group.add_argument(
    '-n-book',
    type=int,
    default=10,
    help='Length of opening lines to use in plies.')
subgroup = group.add_mutually_exclusive_group(required=True)
subgroup.add_argument('-movetime', type=int, help='Time per move in ms')
subgroup.add_argument('-nodes', type=int, help='Nodes per move')

group = parser.add_argument_group('Options to tune')
group.add_argument(
    '-opt',
    nargs='+',
    action='append',
    default=[],
    metavar=(
        'name',
        'lower, upper'),
    help='Integer option to tune.')
group.add_argument(
    '-c-opt',
    nargs='+',
    action='append',
    default=[],
    metavar=(
        'name',
        'value'),
    help='Categorical option to tune')

group = parser.add_argument_group('Optimization parameters')
group.add_argument(
    '-base-estimator',
    default='GP',
    help='One of "GP", "RF", "ET", "GBRT"')
group.add_argument(
    '-n-initial-points',
    type=int,
    default=10,
    help='Number of points chosen before approximating with base estimator.')
group.add_argument(
    '-acq-func',
    default='gp_hedge',
    help='Can be either of "LCB" for lower confidence bound.  "EI" for negative expected improvement.  "PI" for negative probability of improvement.  "gp_hedge" Probabilistically choose one of the above three acquisition functions at every iteration.')
group.add_argument('-acq-optimizer', default='auto', help='Either "sampling" or "lbfgs"')


async def load_engine(engine_args, name, debug=False):
    args = next(a for a in engine_args if a['name'] == name)
    curdir = str(pathlib.Path(__file__).parent.parent)
    popen_args = {}
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
        if not path.is_file():
            print('Unable to locate engines.json file.')
            return
        return json.load(open(str(path)))
    else:
        return json.load(open(conf))


async def get_value(enginea, engineb, args, book, limit):
    # We configure enginea. Engineb is simply our opponent
    enginea.id['args'] = args
    engineb.id['args'] = {}
    await enginea.configure(args)
    score = 0
    games = []
    init_board = board = random.choice(book)
    for i in range(2):
        white, black = (enginea, engineb) if i % 2 == 0 else (engineb, enginea)
        board = init_board.copy()
        # TODO: Stop games earlier when big advantage or low score for long
        for ply in range(int(board.turn == chess.BLACK), 160):
            if not board.is_game_over(claim_draw=True):
                play = await (white, black)[ply % 2].play(board, limit, game=i)
                board.push(play.move)
        result = board.result(claim_draw=True)
        if result == '1-0' and i % 2 == 0 or result == '0-1' and i % 2 == 1:
            score += 1
        if result == '1-0' and i % 2 == 1 or result == '0-1' and i % 2 == 0:
            score -= 1
        game = chess.pgn.Game.from_board(board)
        game.headers.update({
            'Event': 'Tuning',
            'White': white.id['name'],
            'WhiteArgs': repr(white.id['args']),
            'Black': black.id['name'],
            'BlackArgs': repr(black.id['args']),
            'Round': i
        })
        games.append(game)
    return games, score


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


async def run(opt, engines, book, limit, dim_names, concurrency, n_games, options):
    completed = 0
    started = 0
    queue = asyncio.Queue()

    def start_new(conc_id, x=None):
        nonlocal started
        if started >= n_games:
            return
        started += 1
        if x is None:
            x = opt.ask()
        args = x_to_args(x, dim_names, options)
        print(f'Starting game {started}/{n_games} with {args}')
        enginea, engineb = engines[conc_id]
        task = asyncio.create_task(get_value(
            enginea, engineb, args, book, limit))
        task.add_done_callback(functools.partial(on_done, x, conc_id))

    def on_done(x, conc_id, task):
        if task.cancelled():
            logging.debug('Task was cancelled.')
            queue.put_nowait(None)
            # We only get cancelled when we're supposed to shut down,
            # so no reason to start a new loop.
            return
        elif task.exception():
            print('Error while excecuting game')
            task.print_stack()
            queue.put_nowait(None)
        else:
            games, y = task.result()
            opt.tell(x, -y)  # opt is minimizing
            queue.put_nowait((games, x, y))

        start_new(conc_id)

    # If all threads call opt.ask at the same time, we risk them all getting
    # the same response. Hence we coordinate the initals asks.
    for conc_id, x_init in enumerate(opt.ask(concurrency)):
        start_new(conc_id, x_init)

    try:
        for _ in range(n_games):
            res = await queue.get()
            if res:
                yield res
    except asyncio.CancelledError:
        print('Cancelled')
        return


async def main():
    args = parser.parse_args()

    book = []
    if args.book:
        with open(args.book, encoding='latin-1') as file:
            for game in iter((lambda: chess.pgn.read_game(file)), None):
                board = game.board()
                for _, move in zip(range(args.n_book), game.mainline_moves()):
                    board.push(move)
                book.append(board)
        print(f'Loaded book with {len(book)} positions')
    else:
        book.append(chess.Board())
        print('Starting every game from initial position')

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    print('Loading engines')
    conf = load_conf(args.conf)
    engines = await asyncio.gather(*(asyncio.gather(
        load_engine(conf, args.engine),
        load_engine(conf, args.opp_engine or args.engine))
        for _ in range(args.concurrency)))
    options = engines[0][0].options

    print('Parsing options')
    dim_names = []
    dimensions = []
    for name, *lower_upper in args.opt:
        opt = options.get(name)
        if not opt:
            print(f'Error: engine has no option {name}')
            continue
        dim_names.append(name)
        lower, upper = map(int, lower_upper) if lower_upper else (opt.min, opt.max)
        dimensions.append(skopt.utils.Integer(lower, upper, name=name))
    for name, *categories in args.c_opt:
        opt = options.get(name)
        if not opt:
            print(f'Error: engine has no option {name}')
            continue
        dim_names.append(name)
        cats = categories or opt.var
        cats = [opt.var.index(cat) for cat in cats]
        dimensions.append(skopt.utils.Categorical(cats, name=name))
    if not dimensions:
        print('Warning: No options specified for tuning.')

    opt = skopt.Optimizer(
        dimensions,
        base_estimator=args.base_estimator,
        n_initial_points=args.n_initial_points,
        acq_func=args.acq_func,
        acq_optimizer=args.acq_optimizer,
        #acq_func_kwargs={'noise': 1}
    )

    if args.games_file:
        games_file = args.games_file.open('a')
    else:
        games_file = sys.stdout

    cached_games = 0
    if args.log_file:
        print(f'Reading {args.log_file}')
        ahash = hashlib.sha256(repr(args).encode()).hexdigest()
        if args.log_file.is_file():
            with args.log_file.open('r') as file:
                for line in file:
                    obj = json.loads(line)
                    if obj['ahash'] == ahash:
                        x, y = obj['x'], obj['y']
                        print(f'Using {x} => {y} from log-file')
                        try:
                            # Don't fit the model yet, since we aren't asking
                            opt.tell(x, -y, fit=False)
                            cached_games += 1
                        except ValueError as e:
                            print('Ignoring bad data point', e)
        log_file = args.log_file.open('a')
    else:
        log_file = None

    limit = chess.engine.Limit(
        nodes=args.nodes,
        time=args.movetime and args.movetime / 1000)

    async for games, x, y in run(opt, engines, book, limit, dim_names, args.concurrency, args.n - cached_games, options):
        for game in games:
            print(game, file=games_file, flush=True)
        print(x, '=>', y)
        if log_file:
            x = [xi if type(xi) == str else float(xi) for xi in x]
            y = float(y)
            print(json.dumps({'ahash': ahash, 'x': x, 'y': y}), file=log_file, flush=True)

    if opt.Xi and opt.models:
        print('Summarizing best values')
        X = np.vstack((
            opt.space.rvs(n_samples=args.n**2),
            opt.Xi))
        Xt = opt.space.transform(X)
        y_pred, sigma = opt.models[-1].predict(Xt, return_std=True)
        y_pred = -y_pred # Change to maximization
        for kappa in range(4):
            i = np.argmax(y_pred - kappa * sigma)

            def score_to_elo(score):
                return - 400 * math.log10(2 / (score + 1) - 1)
            elo = score_to_elo(y_pred[i] / 2)
            pm = max(abs(score_to_elo(y_pred[i] / 2 + sigma[i] / 2) - elo),
                     abs(score_to_elo(y_pred[i] / 2 + sigma[i] / 2) - elo))
            print(f'Best expectation (κ={kappa}): {X[i]}'
                  f' = {y_pred[i]/2:.3f} ± {sigma[i]/2:.3f}'
                  f' (ELO-diff {elo:.3f} ± {pm:.3f})')

        if len(dimensions) == 1:
            plot_optimizer(opt, dimensions[0].low, dimensions[0].high)
    else:
        print('Not enought data to summarize results.')

    logging.debug('Quitting engines')
    await asyncio.gather(*(e.quit() for es in engines for e in es))

asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
try:
    asyncio.run(main())
except KeyboardInterrupt:
    logging.debug('KeyboardInterrupt at root')
