from collections import namedtuple
import time
import numpy as np
import random

import mcts

Stats = namedtuple('Stats', ['kl_div', 'rolls', 'elapsed'])

class MCTS_Controller:
    def __init__(self, fasttext_model, uci_format=False):
        self.model = mcts.Model(fasttext_model)
        self.node = None
        self.uci_format = uci_format
        self.should_stop = False

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
                if self.uci_format:
                    san = node.move.uci()
                pv.append(san)

            if self.uci_format:
                print(f'multipv {i+1}', ' '.join(pv))
            else:
                if len(pv) >= 10:
                    pv = pv[:10] + ['...']
                print(f'Pv{i+1}:', ', '.join(pv))

    def print_stats(self, is_first):
        if is_first:
            self.old_dist = np.array([1 + n.N for n in self.node.children])
            self.old_dist = self.old_dist / self.old_dist.sum()
            self.start_time = time.time()
            self.old_time = time.time()
            if not self.uci_format:
                print()  # Make space
            return 1
        else:
            dist = np.array([1 + n.N for n in self.node.children])
            dist = dist / dist.sum()
            kl_div = np.sum(dist * np.log(dist / self.old_dist))
            self.old_dist = dist
            new_time = time.time()
            nps = 100 / (new_time - self.old_time)
            self.old_time = new_time
            t = new_time - self.start_time
            if self.uci_format:
                print(f'info score {self.node.Q} time {t} nodes {self.node.N}')
            else:
                print(f'KL: {kl_div:.3} rolls: {self.node.N}'
                      f' nps: {nps:.0f} t: {t:.1f}s')
            return kl_div

    def stop(self):
        self.should_stop = True

    def find_move(self, board, min_kldiv=0, max_rolls=0, max_time=0,
                        pvs=0, debug=False, temperature=False):
        """ Searches until kl_div is below `min_kldiv` or for `movetime' milliseconds, or if 0, for `rolls` rollouts. """
        # We try to reuse the previous node, but if we can't, we create a new one.
        if self.node:
            # Check if the board is at one of our children (cheap pondering)
            for node in self.node.children:
                if node.board == board:
                    self.node = node
                    if debug:
                        print('info string Reusing node from ponder.')
                    break

        # If we weren't able to find the board, make a new node
        if not self.node or self.node.board != board:
            self.node = mcts.Node(board, None, 0, self.model)
            if debug:
                print('info string Creating new root node.')

        # Print priors for new root node
        if debug:
            self.node.rollout()  # Ensure children are expanded
            nodes = sorted(self.node.children, key=lambda n: n.P, reverse=True)[:7]
            if not self.uci_format:
                print('Priors:', ', '.join(
                    f'{board.san(n.move)} {n.P:.1%}' for n in nodes))

        # Find move to play
        first = True
        kl_div = 1
        rolls = 0
        start_time = time.time()
        while True:
            rolls += 1
            self.node.rollout()
            if self.node.N % 100 == 0:
                # Remove old PVs and stats lines
                real_pvs = min(pvs, len(self.node.children))
                if not first and not self.uci_format:
                    print(f"\u001b[1A\u001b[K" * (real_pvs + 1), end='')
                if pvs:
                    self.print_pvs(real_pvs)
                kl_div = self.print_stats(first)
                if self.should_stop:
                    self.should_stop = False
                    break
                if max_time > 0 and time.time() > start_time + max_time:
                    break
                if max_rolls > 0 and rolls >= max_rolls:
                    break
                if min_kldiv > 0 and kl_div < min_kldiv:
                    break
                first = False

        # Pick best or random child
        if temperature:
            counts = [(n.N / self.node.N)**(1 / temperature) for n in self.node.children]
            node = random.choices(self.node.children, weights=counts)[0]
            if debug:
                o = sorted(self.node.children, key=lambda n: -n.N).index(node)
                ordinal = (lambda n: "%d%s" % (n, "tsnrhtdd"[
                           (n / 10 % 10 != 1) * (n % 10 < 4) * n % 10::4]))(o + 1)
                #pct = counts[self.node.children.index(node)] * 100
                if not self.uci_format:
                    print(f'Chose {ordinal} child. (temp={temperature})')
            self.node = node
        else:
            self.node = max(self.node.children, key=lambda n: n.N)

        stats = Stats(kl_div, rolls, time.time() - start_time)
        return self.node, stats

