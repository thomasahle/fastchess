import asyncio
import itertools
import random
import logging

import chess.pgn
import chess.engine
import chess
from chess import WHITE, BLACK


class Arena:
    MATE_SCORE = 32767

    def __init__(self, enginea, engineb, limit, max_len, win_adj_count, win_adj_score):
        self.enginea = enginea
        self.engineb = engineb
        self.limit = limit
        self.max_len = max_len
        self.win_adj_count = win_adj_count
        self.win_adj_score = win_adj_score

    def adjudicate(self, score_hist):
        if len(score_hist) > self.max_len:
            return '1/2-1/2'
        # Note win_adj_count is in moves, not plies
        count_max = self.win_adj_count * 2
        if count_max > len(score_hist):
            return None
        # Test if white has been winning. Notice score_hist is from whites pov.
        if all(v >= self.win_adj_score for v in score_hist[-count_max:]):
            return '1-0'
        # Test if black has been winning
        if all(v <= -self.win_adj_score for v in score_hist[-count_max:]):
            return '0-1'
        return None

    async def play_game(self, init_node, game_id, white, black):
        """ Yields (play, error) tuples. Also updates the game with headers and moves. """
        try:
            game = init_node.root()
            node = init_node
            score_hist = []
            for ply in itertools.count(int(node.board().turn == BLACK)):
                board = node.board()

                adj_result = self.adjudicate(score_hist)
                if adj_result is not None:
                    game.headers.update({
                        'Result': adj_result,
                        'Termination': 'adjudication'
                    })
                    return

                if board.is_game_over(claim_draw=True):
                    result = board.result(claim_draw=True)
                    game.headers["Result"] = result
                    return

                # Try to actually make a move
                play = await (white, black)[ply % 2].play(
                    board, self.limit, game=game_id,
                    info=chess.engine.INFO_BASIC | chess.engine.INFO_SCORE)
                yield play, None

                if play.resigned:
                    game.headers.update({'Result': ('0-1', '1-0')[ply % 2]})
                    node.comment += f'; {("White","Black")[ply%2]} resgined'
                    return

                node = node.add_variation(play.move, comment=
                        f'{play.info.get("score",0)}/{play.info.get("depth",0)}'
                        f' {play.info.get("time",0)}s')

                # Adjudicate game by score, save score in wpov
                try:
                    score_hist.append(play.info['score'].white().score(
                        mate_score=max(self.win_adj_score, Arena.MATE_SCORE)))
                except KeyError:
                    logging.debug('Engine didn\'t return a score for adjudication. Assuming 0.')
                    score_hist.append(0)

        except (asyncio.CancelledError, KeyboardInterrupt) as e:
            print('play_game Cancelled')
            # We should get CancelledError when the user pressed Ctrl+C
            game.headers.update({'Result': '*', 'Termination': 'unterminated'})
            node.comment += '; Game was cancelled'
            await asyncio.wait([white.quit(), black.quit()])
            yield None, e
        except chess.engine.EngineError as e:
            game.headers.update(
                {'Result': ('0-1', '1-0')[ply % 2], 'Termination': 'error'})
            node.comment += f'; {("White","Black")[ply%2]} terminated: {e}'
            yield None, e

    async def configure(self, args):
        # We configure enginea, engineb is our unchanged opponent.
        # Maybe this should be refactored.
        self.enginea.id['args'] = args
        self.engineb.id['args'] = {}
        try:
            await self.enginea.configure(args)
        except chess.engine.EngineError as e:
            print(f'Unable to start game {e}')
            return [], 0

    async def run_games(self, book, game_id=0, games_played=2):
        score = 0
        games = []
        assert games_played % 2 == 0
        for r in range(games_played//2):
            init_board = random.choice(book)
            for color in [WHITE, BLACK]:
                white, black = (self.enginea, self.engineb) if color == WHITE \
                    else (self.engineb, self.enginea)
                game_round = games_played * game_id + color + 2*r
                game = chess.pgn.Game({
                    'Event': 'Tune.py',
                    'White': white.id['name'],
                    'WhiteArgs': repr(white.id['args']),
                    'Black': black.id['name'],
                    'BlackArgs': repr(black.id['args']),
                    'Round': game_round
                })
                games.append(game)
                # Add book moves
                game.setup(init_board.root())
                node = game
                for move in init_board.move_stack:
                    node = node.add_variation(move, comment='book')
                # Run engines
                async for _play, er in self.play_game(node, game_round, white, black):
                    # If an error occoured, return as much as we got
                    if er is not None:
                        return games, score, er
                result = game.headers["Result"]
                if result == '1-0' and color == WHITE or result == '0-1' and color == BLACK:
                    score += 1
                if result == '1-0' and color == BLACK or result == '0-1' and color == WHITE:
                    score -= 1
        return games, score/games_played, None
