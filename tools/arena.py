import asyncio
import itertools

import chess.pgn

class Arena:
    def __init__(self, enginea, engineb, limit, max_len):
        self.enginea = enginea
        self.engineb = engineb
        self.limit = limit
        self.max_len = max_len

    async def play_game(self, init_node, game_id, flip=False):
        """ Yields (play, error) tupples. Also updates the game with headers and moves. """
        white, black = (self.enginea, self.engineb) if not flip else (self.engineb, self.enginea)
        try:
            game = init_node.root()
            node = init_node
            for ply in itertools.count(int(node.board().turn == chess.BLACK)):
                board = node.board()
                # TODO: Add options for stopping games earlier when a player has a
                #       big advantage or low score for long.
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
                    board, self.limit, game = game_id,
                    info = chess.engine.INFO_BASIC | chess.engine.INFO_SCORE)
                yield play, None

                if play.resigned:
                    game.headers.update({'Result': ('0-1', '1-0')[ply % 2]})
                    node.comment += f'; {("White","Black")[ply%2]} resgined'
                    return

                node = node.add_variation(play.move, comment=
                        f'{play.info.get("score",0)}/{play.info.get("depth",0)}'
                        f' {play.info.get("time",0)}s')

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
        # We configure enginea. Engineb is simply our opponent.
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
            white, black = (self.enginea, self.engineb) if i % 2 == 0 else (self.engineb, self.enginea)
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
            async for _play, er in self.play_game(node, games_played * game_id + i, flip=i%2):
                # If an error occoured, return as much as we got
                if er is not None:
                    return games, score, er
            result = game.headers["Result"]
            if result == '1-0' and i % 2 == 0 or result == '0-1' and i % 2 == 1:
                score += 1
            if result == '1-0' and i % 2 == 1 or result == '0-1' and i % 2 == 0:
                score -= 1
        return games, score, None

