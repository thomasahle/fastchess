import asyncio
import itertools

import chess.pgn


class Arena:
    MATE_SCORE = 32767

    def __init__(self, enginea, engineb, limit, max_len, win_adj_count, win_adj_score):
        self.enginea = enginea
        self.engineb = engineb
        self.limit = limit
        self.max_len = max_len
        self.win_adj_count = win_adj_count
        self.win_adj_score = win_adj_score

    def adjudicate(self, score_hist, is_tuned_eng_white):
        res_value, result, w_good_cnt, b_good_cnt = 0, None, 0, 0
        k = len(score_hist) - 1
        count_max = self.win_adj_count * 2

        if k + 1 < count_max or abs(score_hist[k]) < self.win_adj_score:
            return res_value, result

        for i in itertools.count(k, step=-1):
            if i <= k - count_max:
                break
            if score_hist[i] >= self.win_adj_score:
                w_good_cnt += 1
            elif score_hist[i] <= -self.win_adj_score:
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

    async def play_game(self, init_node, game_id, flip=False):
        """ Yields (play, error) tuples. Also updates the game with headers and moves. """
        white, black = (
            self.enginea, self.engineb) if not flip else (
            self.engineb, self.enginea)
        try:
            game = init_node.root()
            node = init_node
            score_hist = []
            for ply in itertools.count(int(node.board().turn == chess.BLACK)):
                board = node.board()
                if ply > self.max_len:
                    game.headers.update({
                        'Result': '1/2-1/2',
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

                node = node.add_variation(play.move, comment=f'{play.info.get("score",0)}/{play.info.get("depth",0)}'
                                          f' {play.info.get("time",0)}s')

                # Adjudicate game by score, save score in wpov
                try:
                    score_hist.append(play.info['score'].white().score(
                        mate_score=max(self.win_adj_score, Arena.MATE_SCORE)))
                except KeyError:
                    score_hist.append(0)
                except Exception:
                    score_hist.append(0)
                res_value, result = self.adjudicate(score_hist, not flip)
                if res_value != 0:
                    game.headers.update({
                        'Result': result,
                        'Termination': 'adjudication based on score'
                    })
                    return
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

    async def run_games(self, init_board, game_id=0, games_played=2):
        score = 0
        games = []
        for i in range(games_played):
            white, black = (self.enginea, self.engineb) if i % 2 == 0 else (
                self.engineb, self.enginea)
            game = chess.pgn.Game({
                'Event': 'Tune.py',
                'White': white.id['name'],
                'WhiteArgs': repr(white.id['args']),
                'Black': black.id['name'],
                'BlackArgs': repr(black.id['args']),
                'Round': games_played * game_id + i
            })
            games.append(game)
            # Add book moves
            game.setup(init_board.root())
            node = game
            for move in init_board.move_stack:
                node = node.add_variation(move, comment='book')
            # Run engines
            async for _play, er in self.play_game(node, games_played * game_id + i, flip=i % 2):
                # If an error occoured, return as much as we got
                if er is not None:
                    return games, score, er
            result = game.headers["Result"]
            if result == '1-0' and i % 2 == 0 or result == '0-1' and i % 2 == 1:
                score += 1
            if result == '1-0' and i % 2 == 1 or result == '0-1' and i % 2 == 0:
                score -= 1
        return games, score, None
