import sys
import chess
import chess.engine
import pathlib


def new_engine():
    #python = '/usr/local/bin/python3'
    python = sys.executable
    d = pathlib.Path(__file__).parent.parent
    args = [python, '-u', str(d / 'uci.py'), str(d / 'model.bin'), '-occ', '-debug']
    return chess.engine.SimpleEngine.popen_uci(args, debug=True)


limits = [
    chess.engine.Limit(time=1),
    chess.engine.Limit(nodes=1000),
    chess.engine.Limit(white_clock=1, black_clock=1),
    chess.engine.Limit(
        white_clock=1,
        black_clock=1,
        white_inc=1,
        black_inc=1,
        remaining_moves=2)
]
for limit in limits:
    engine = new_engine()
    board = chess.Board()
    while not board.is_game_over() and len(board.move_stack) < 3:
        result = engine.play(board, limit)
        board.push(result.move)
    engine.quit()
