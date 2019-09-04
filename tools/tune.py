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
import itertools
from collections import namedtuple

import chess.pgn
import chess.engine
import chess
import numpy as np
import skopt


MATE_SCORE = 32767

warnings.filterwarnings(
    'ignore',
    message='The objective has been evaluated at this point before.')

parser = argparse.ArgumentParser()
parser.add_argument('-debug', nargs='?', metavar='PATH', const=sys.stdout,
                    default=None, type=pathlib.Path,
                    help='Enable debugging of engines.')
parser.add_argument('-log-file', type=pathlib.Path,
                    help='Used to recover from crashes')
parser.add_argument('-n', type=int, default=100,
                    help='Number of iterations')
parser.add_argument('-concurrency', type=int, default=1,
                    help='Number of concurrent games')
parser.add_argument('-games-file', type=pathlib.Path,
                    help='Store all games to this pgn')

group = parser.add_argument_group('Engine options')
group.add_argument('engine',
                   help='Engine to tune')
group.add_argument('-conf',
                   help='Engines.json file to load from')
group.add_argument('-opp-engine',
                   help='Tune against a different engine')

group = parser.add_argument_group('Games format')
group.add_argument('-book', type=pathlib.Path,
                   help='pgn file with opening lines.')
group.add_argument('-n-book', type=int, default=10,
                   help='Length of opening lines to use in plies.')
group.add_argument('-max-len', type=int, default=10000,
                   help='Maximum length of game in plies before termination.')
subgroup = group.add_mutually_exclusive_group(required=True)
subgroup.add_argument('-movetime', type=int,
                      help='Time per move in ms')
subgroup.add_argument('-nodes', type=int,
                      help='Nodes per move')

group = parser.add_argument_group('Options to tune')
group.add_argument('-opt', nargs='+', action='append', default=[],
                   metavar=('NAME', 'LOWER, UPPER'),
                   help='Integer option to tune.')
group.add_argument('-c-opt', nargs='+', action='append', default=[],
                   metavar=('NAME', 'VALUE'),
                   help='Categorical option to tune')

group = parser.add_argument_group('Optimization parameters')
group.add_argument('-base-estimator', default='GP',
                   help='One of "GP", "RF", "ET", "GBRT"')
group.add_argument('-n-initial-points', type=int, default=10,
                   help='Number of points chosen before approximating with base estimator.')
group.add_argument('-acq-func', default='gp_hedge',
                   help='Can be either of "LCB" for lower confidence bound.'
                   ' "EI" for negative expected improvement.'
                   ' "PI" for negative probability of improvement.'
                   ' "gp_hedge" (default) Probabilistically choose one of the above'
                   ' three acquisition functions at every iteration.')
group.add_argument('-acq-optimizer', default='auto',
                   help='Either "sampling" or "lbfgs"')
group.add_argument('-acq-noise', default='gaussian', metavar='VAR',
                   help='For the Gaussian Process optimizer, use this to specify the'
                   ' variance of the assumed noise. Larger values means more exploration.')

group = parser.add_argument_group('Adjudication options')
group.add_argument('-adj-count', type=int, default=4,
                   help='Count of successive moves, default=4')
group.add_argument('-adj-score', type=int, default=400,
                   help='Score in cp, default=400')


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


def adjudicate(score_hist, is_tuned_eng_white, adj_count, adj_score):
    """
    :score_hist: A list of wpov scores.
    :is_tuned_eng_white: The tuned engine is playing white.
    """
    res_value, result, w_good_cnt, b_good_cnt = 0, None, 0, 0
    k = len(score_hist) - 1
    count_max = adj_count * 2
    
    if k + 1 < count_max or abs(score_hist[k]) < adj_score:
        return res_value, result    
    
    for i in itertools.count(k, step=-1):
        if i <= k - count_max:
            break
        if score_hist[i] >= adj_score:
            w_good_cnt += 1
        elif score_hist[i] <= -adj_score:
            b_good_cnt += 1

    if w_good_cnt >= count_max or b_good_cnt >= count_max:
        res_value = -1
        if w_good_cnt >= count_max and is_tuned_eng_white:
            res_value = 1
        elif b_good_cnt >= count_max and not is_tuned_eng_white:
            res_value = 1

        if res_value > 0:
            result = '1-0' if is_tuned_eng_white else '0-1'
        else:
            result = '1-0' if not is_tuned_eng_white else '0-1'

    return res_value, result


Context = namedtuple('Context', 'enginea engineb limit max_len')
GAMES_PLAYED = 2  # Number of games played with a given book


async def get_value(context, init_board, args, game_id, adj_count, adj_score):
    enginea, engineb, limit, max_len = context
    # We configure enginea. Engineb is simply our opponent
    enginea.id['args'] = args
    engineb.id['args'] = {}
    try:
        await enginea.configure(args)
    except chess.engine.EngineError as e:
        print(f'Unable to start game {e}')
        return [], 0
    score = 0
    games = []
    error = None
    for i in range(GAMES_PLAYED):
        score_hist = []
        white, black = (enginea, engineb) if i % 2 == 0 else (engineb, enginea)
        game = chess.pgn.Game({
            'Event': 'Tune.py',
            'White': white.id['name'],
            'WhiteArgs': repr(white.id['args']),
            'Black': black.id['name'],
            'BlackArgs': repr(black.id['args']),
            'Round': GAMES_PLAYED * game_id + i
        })
        games.append(game)
        # Add book moves
        game.setup(init_board.root())
        node = game
        for move in init_board.move_stack:
            node = node.add_variation(move, comment='book')
            score_hist.append(0)
        # Run engines
        try:
            for ply in itertools.count(int(node.board().turn == chess.BLACK)):
                board = node.board()
                if ply > max_len:
                    game.headers.update({
                        'Result': '1/2-1/2',
                        'Termination': 'adjudication'
                    })
                    break
                if board.is_game_over(claim_draw=True):
                    result = board.result(claim_draw=True)
                    if result == '1-0' and i % 2 == 0 or result == '0-1' and i % 2 == 1:
                        score += 1
                    if result == '1-0' and i % 2 == 1 or result == '0-1' and i % 2 == 0:
                        score -= 1
                    game.headers["Result"] = result
                    break
                # Try to actually make a move
                play = await (white, black)[ply % 2].play(
                    board, limit, game=GAMES_PLAYED * game_id + i,
                    info=chess.engine.INFO_BASIC | chess.engine.INFO_SCORE)
                if play.resigned:
                    game.headers.update({'Result': ('0-1', '1-0')[ply % 2]})
                    node.comment += f'; {("White","Black")[ply%2]} resgined'
                    score += -1 if (i + ply) % 2 == 0 else 1
                    break
                node = node.add_variation(play.move, comment=f'{play.info.get("score",0)}/{play.info.get("depth",0)}'
                                          f' {play.info.get("time",0)}s')
                # Adjudicate game by score, save score in wpov
                try:
                    score_hist.append(play.info['score'].white().score(
                            mate_score=max(adj_score, MATE_SCORE)))
                except KeyError:
                    logging.debug(play.info)
                    logging.exception('Engine info line has no score.')
                    score_hist.append(0)
                except Exception:
                    logging.debug(play.info)
                    logging.exception('Unexpected exception')
                    score_hist.append(0)
                res_value, result = adjudicate(score_hist, i % 2 == 0, adj_count, adj_score)
                if res_value != 0:
                    score += res_value
                    game.headers.update({
                        'Result': result,
                        'Termination': 'adjudication based on score'
                        })
                    break
        except (asyncio.CancelledError, KeyboardInterrupt) as e:
            print('get_value Cancelled')
            # We should get CancelledError when the user pressed Ctrl+C
            game.headers.update({'Result': '*', 'Termination': 'unterminated'})
            node.comment += '; Game was cancelled'
            await asyncio.wait([enginea.quit(), engineb.quit()])
            error = e
            break
        except chess.engine.EngineError as e:
            game.headers.update(
                {'Result': ('0-1', '1-0')[ply % 2], 'Termination': 'error'})
            node.comment += f'; {("White","Black")[ply%2]} terminated: {e}'
            score += -1 if ply % 2 == 0 else 1
            # await asyncio.wait([enginea.quit(), engineb.quit()])
            error = e
            break
        except BaseException:
            game.headers.update({'Result': '*', 'Termination': 'error'})
            node.comment += '; Unexpected error'
            await asyncio.wait([enginea.quit(), engineb.quit()])
            raise
    return games, score, error


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
            return - 400 * math.log10(2 / (score + 1) - 1)
        elo = score_to_elo(y_pred[i] / 2)
        pm = max(abs(score_to_elo(y_pred[i] / 2 + sigma[i] / 2) - elo),
                 abs(score_to_elo(y_pred[i] / 2 + sigma[i] / 2) - elo))
        print(f'Best expectation (κ={kappa}): {X[i]}'
              f' = {y_pred[i]/2:.3f} ± {sigma[i]/2:.3f}'
              f' (ELO-diff {elo:.3f} ± {pm:.3f})')


async def main():
    args = parser.parse_args()

    book = []
    if args.book:
        book.extend(load_book(args.book, args.n_book))
        print(f'Loaded book with {len(book)} positions')
    if not book:
        book.append(chess.Board())
        print('No book. Starting every game from initial position.')

    if args.debug:
        if args.debug == sys.stdout:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=args.debug, filemode='w')

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
        acq_func_kwargs={'noise': args.acq_noise}
    )

    if args.games_file:
        games_file = args.games_file.open('a')
    else:
        games_file = sys.stdout

    if args.log_file:
        key_args = args.__dict__.copy()
        # Not all arguments change the results, so no need to keep them in the key.
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

    # Run tasks concurrently
    try:
        started = cached_games

        def on_done(task):
            if task.exception():
                logging.error('Error while excecuting game')
                task.print_stack()

        def new_game(context):
            x = opt.ask()
            engine_args = x_to_args(x, dim_names, options)
            print(f'Starting game {started}/{args.n} with {engine_args}')
            task = asyncio.create_task(
                get_value(
                    context,
                    random.choice(book),
                    engine_args,
                    started,
                    args.adj_count,
                    args.adj_score))
            # We tag the task with some attributes that we need when it finishes.
            setattr(task, 'tune_x', x)
            setattr(task, 'tune_context', context)
            setattr(task, 'tune_game_id', started)
            task.add_done_callback(on_done)
            return task
        tasks = []
        xs = opt.ask(min(args.concurrency, args.n - started)
                     ) if args.n - started > 0 else []
        for conc_id, x_init in enumerate(xs):
            enginea, engineb = engines[conc_id]
            context = Context(enginea, engineb, limit, args.max_len)
            tasks.append(new_game(context))
            started += 1
        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = list(pending)
            for task in done:
                res = task.result()
                context, x, game_id = task.tune_context, task.tune_x, task.tune_game_id
                games, y, er = res
                for game in games:
                    print(game, end='\n\n', file=games_file, flush=True)
                if er:
                    print('Game erred:', er, type(er))
                    continue
                opt.tell(x, -y)  # opt is minimizing
                # Delete old models to save space. Note hat for the first 10
                # tells no model is created, as we sare still just querying at
                # random.
                logging.debug(f'Number of models {len(opt.models)}')
                if len(opt.models) > 1:
                    del opt.models[0]
                results = ', '.join(g.headers['Result'] for g in games)
                print(f'Finished game {game_id} {x} => {y} ({results})')
                if data_logger:
                    data_logger.store(x, y)
                if started < args.n:
                    tasks.append(new_game(context))
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
