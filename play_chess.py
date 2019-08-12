import chess
import re
import fastchess
import argparse
import random
import time
import mcts
import numpy as np
import time


def get_user_move(board):
    # Get well-formated move
    move = None
    while move is None:
        san_option = random.choice([board.san(m) for m in board.legal_moves])
        uci_option = random.choice([m.uci() for m in board.legal_moves])
        uci = input(f'Your move (e.g. {san_option} or {uci_option}): ')
        for parse in (board.parse_san, chess.Move.from_uci):
            try:
                move = parse(uci)
            except ValueError:
                pass

    # Check legality
    if move not in board.legal_moves:
        print('Illegal move.')
        return get_user_move(board)

    return move


def get_user_color():
    color = ''
    while color not in ('white', 'black'):
        color = input('Do you want to be white or black? ')
    return chess.WHITE if color == 'white' else chess.BLACK


def print_unicode_board(board, perspective=chess.WHITE):
    """ Prints the position from a given perspective. """
    uni_pieces = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'}
    sc, ec = '\x1b[0;30;107m', '\x1b[0m'
    for r in range(8) if perspective == chess.BLACK else range(7, -1, -1):
        line = [f'{sc} {r+1}']
        for c in range(8) if perspective == chess.WHITE else range(7, -1, -1):
            color = '\x1b[48;5;255m' if (r + c) % 2 == 1 else '\x1b[48;5;253m'
            if board.move_stack:
                if board.move_stack[-1].to_square == 8*r + c:
                    color = '\x1b[48;5;153m'
                elif board.move_stack[-1].from_square == 8*r + c:
                    color = '\x1b[48;5;153m'
            piece = board.piece_at(8*r + c)
            if piece:
                line.append(color + uni_pieces[piece.symbol()])
            else:
                line.append(color + ' ')
        print(' ' + ' '.join(line) + f' {sc} {ec}')
    if perspective == chess.WHITE:
        print(f' {sc}   a b c d e f g h  {ec}\n')
    else:
        print(f' {sc}   h g f e d c b a  {ec}\n')


def self_play(model, rand=0, debug=False, board=None):
    if not board:
        board = chess.Board()

    while not board.is_game_over():
        print_unicode_board(board)
        move = model.find_move(board, debug=debug, temperature=rand)
        print(f'\n My move: {board.san(move)}')
        board.push(move)

    # Print status
    print_unicode_board(board)
    print('Result:', board.result())


def play(model, rand=0, debug=False, board=None):
    user_color = get_user_color()
    if not board:
        board = chess.Board()

    while not board.is_game_over():
        print_unicode_board(board, perspective=user_color)
        if user_color == board.turn:
            move = get_user_move(board)
        else:
            move = model.find_move(board, debug=debug, temperature=rand)
            print(f' My move: {board.san(move)}')
        board.push(move)

    # Print status
    print_unicode_board(board, perspective=user_color)
    print('Result:', board.result())


class MCTS_Model:
    def __init__(self, fasttext_model, rolls, pvs=0):
        self.model = mcts.Model(fasttext_model)
        self.rolls = rolls
        self.pvs = pvs
        self.node = None

    def print_pvs(self, pvs):
        """ print `pvs` pvs starting from root """
        root = self.node
        for i in range(pvs):
            pv = []
            node = root
            while node.children:
                if node == root:
                    node = sorted(node.children, key=lambda n: -n.N)[i]
                    san = node.parent_board.san(node.move)
                    san += f' {node.N/root.N:.1%} ({float(-node.Q):.2})'
                else:
                    node = max(node.children, key=lambda n: n.N)
                    san = node.parent_board.san(node.move)
                pv.append(san)
            # Trim length of pv
            if len(pv) >= 10:
                pv = pv[:10] + ['...']
            print(f'Pv{i+1}:', ', '.join(pv))

    def print_stats(self, is_first):
        if is_first:
            self.old_dist = np.array([1+n.N for n in self.node.children])
            self.old_dist = self.old_dist/self.old_dist.sum()
            self.start_time = time.time()
            self.old_time = time.time()
            print()  # Make space
            return 1
        else:
            dist = np.array([1+n.N for n in self.node.children])
            dist = dist/dist.sum()
            kl_div = np.sum(dist * np.log(dist / self.old_dist))
            self.old_dist = dist
            new_time = time.time()
            nps = 100/(new_time - self.old_time)
            self.old_time = new_time
            t = new_time - self.start_time
            print(f'KL: {kl_div:.3} rolls: {self.node.N}'
                  f' nps: {nps:.0f} t: {t:.1f}s')
            return kl_div

    def find_move(self, board, debug=False, temperature=False):
        # We try to reuse the previous node, but if we can't, we create a new one.
        if self.node:
            # Check if the board is at one of our children (like pondering)
            for node in self.node.children:
                if node.board == board:
                    self.node = node
                    if debug:
                        print('Reusing node from ponder.')
                    break

        # If we weren't able to find the board, make a new node
        if not self.node or self.node.board != board:
            self.node = mcts.Node(board, None, 0, self.model)
            if debug:
                print('Creating new root node.')

        # Print priors for new root node
        if debug:
            self.node.rollout()  # Ensure children are expanded
            nodes = sorted(self.node.children,
                           key=lambda n: n.P, reverse=True)[:7]
            print('Priors:', ', '.join(
                f'{board.san(n.move)} {n.P:.1%}' for n in nodes))

        # Find move to play
        first = True
        while True:
            self.node.rollout()
            if self.node.N % 100 == 0:
                # Remove old PVs and stats lines
                pvs = min(self.pvs, len(self.node.children))
                if not first:
                    print(f"\u001b[1A\u001b[K"*(pvs+1), end='')
                if self.pvs:
                    self.print_pvs(pvs)
                kl_div = self.print_stats(first)
                if kl_div < 1/self.rolls:
                    break
                first = False

        # Pick best or random child
        if temperature:
            counts = [(n.N/self.node.N)**(1/temperature) for n in self.node.children]
            node = random.choices(self.node.children, weights=counts)[0]
            if debug:
                o = sorted(self.node.children, key=lambda n: -n.N).index(node)
                ordinal = (lambda n: "%d%s" % (n, "tsnrhtdd"[
                           (n/10 % 10 != 1)*(n % 10 < 4)*n % 10::4]))(o+1)
                #pct = counts[self.node.children.index(node)] * 100
                print(f'Chose {ordinal} child. (temp={temperature})')
            self.node = node
        else:
            self.node = max(self.node.children, key=lambda n: n.N)

        return self.node.move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Location of fasttext model to use')
    parser.add_argument('-selfplay', action='store_true',
                        help='Play against itself')
    parser.add_argument('-rand', nargs='?', help='Play random moves from the posterior distribution to the 1/temp power.',
                        metavar='TEMP', const=1, default=0, type=float)
    parser.add_argument('-debug', action='store_true',
                        help='Print all predicted labels')
    parser.add_argument('-mcts', nargs='?', help='Play stronger (hopefully)',
                        metavar='ROLLS', const=800, default=1, type=int)
    parser.add_argument(
        '-pvs', nargs='?', help='Show Principal Variations (when mcts)', const=3, default=0, type=int)
    parser.add_argument('-fen', help='Start from given position',
                        default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    parser.add_argument('-occ', action='store_true', help='Add -Occ features')
    parser.add_argument('-profile', action='store_true', help='Run profiling. (Only with selfplay)')
    args = parser.parse_args()

    if args.debug:
        print('Loading model...')
    fastchess_model = fastchess.Model(args.model_path, occ=args.occ)
    model = MCTS_Model(fastchess_model, rolls=args.mcts, pvs=args.pvs)
    board = chess.Board(args.fen)

    try:
        if args.selfplay:
            if args.profile:
                import cProfile as profile
                profile.runctx('self_play(model, rand=args.rand, debug=args.debug, board=board)', globals(), locals())
            else:
                self_play(model, rand=args.rand, debug=args.debug, board=board)
        else:
            play(model, rand=args.rand, debug=args.debug, board=board)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nGoodbye!')


if __name__ == '__main__':
    main()
