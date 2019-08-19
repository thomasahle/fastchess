from collections import namedtuple
import time
import numpy as np
import random
import math
import itertools

import fastchess
import mcts

STAT_INTERVAL = 100
MIN_PV_VISITS = 100

Stats = namedtuple('Stats', ['kl_div', 'rolls', 'elapsed'])


class MCTS_Controller:
    def __init__(self, args):
        self.args = args
        self.node = None
        self.should_stop = False

    def print_stats(self, is_first, pvs):
        if is_first:
            self.old_dist = np.array([1 + n.N for n in self.node.children])
            self.old_dist = self.old_dist / self.old_dist.sum()
            self.start_time = time.time()
            self.old_time = time.time()
            return 1

        dist = np.array([1 + n.N for n in self.node.children])
        dist = dist / dist.sum()
        kl_div = np.sum(dist * np.log(dist / self.old_dist))
        self.old_dist = dist

        new_time = time.time()
        nps = STAT_INTERVAL / (new_time - self.old_time)
        self.old_time = new_time
        t = new_time - self.start_time
        if kl_div > 0:
            print(f'info string kl {-math.log(kl_div):.1f} root_score {self.node.Q}')
        else:
            print(f'info string kl -inf root_score {self.node.Q}')

        root = self.node
        real_pvs = min(pvs, len(root.children))
        root_children = sorted(root.children, key=lambda n: -n.N)
        for i in range(real_pvs):
            node = root_children[i]
            pv = [node.move.uci()]
            Q = node.Q
            N = node.N
            while node.children:
                node = max(node.children, key=lambda n: n.N)
                if node.N < MIN_PV_VISITS:
                    break
                pv.append(node.move.uci())

            score = fastchess.win_to_cp(Q)
            extras = f'time {t*1000:.0f} nodes {self.node.N} nps {nps:.0f}' if i == 0 else ''
            print(
                f'info multipv {i+1} score cp {score:.0f} depth {len(pv)} {extras} pv {" ".join(pv)} string pv_nodes {N}')
            # string pv_score  {score:.0f} pv_nodes {N}')
        return kl_div

    def stop(self):
        self.should_stop = True

    def find_move(self, board, min_kldiv=0, max_rolls=0, max_time=0,
                  pvs=0, temperature=False, use_mcts=True):
        """ Searches until kl_div is below `min_kldiv` or for `movetime' milliseconds, or if 0, for `rolls` rollouts. """
        # We try to reuse the previous node, but if we can't, we create a new one.
        if self.node:
            # Check if the board is at one of our children (cheap pondering)
            for node in self.node.children:
                if node.board == board:
                    self.node = node
                    if self.args.debug:
                        print('info string Reusing node from ponder.')
                    break

        # If we weren't able to find the board, make a new node
        if not self.node or self.node.board != board:
            vec = self.args.model.from_scratch(board)
            self.node = mcts.Node(board, vec, None, 0, self.args)
            if self.args.debug:
                print('info string Creating new root node.')

        # If the node is a reused node and at a repeated position, it will think
        # the game is over, but we still want it to continue playing.
        self.node.game_over = False

        # Print priors for new root node.
        while self.node.N < 2:
            # Ensure children are expanded
            self.node.rollout()
        nodes = sorted(self.node.children, key=lambda n: n.P, reverse=True)[:7]
        print('info string priors', ', '.join(
            f'{board.san(n.move)} {n.P:.1%}' for n in nodes))

        # Find move to play
        self.should_stop = False
        kl_div = 1
        rolls = 0
        start_time = time.time()
        if use_mcts:
            first = True
            for i in itertools.count():
                rolls += 1
                self.node.rollout()
                if i % STAT_INTERVAL == 0:
                    kl_div = self.print_stats(first, pvs)
                    if self.should_stop:
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
            if use_mcts:
                counts = [(n.N / self.node.N)**(1 / temperature)
                          for n in self.node.children]
            else:
                counts = [n.P**(1 / temperature) for n in self.node.children]
            node = random.choices(self.node.children, weights=counts)[0]
            if self.args.debug:
                o = sorted(self.node.children, key=lambda n: -n.N).index(node)
                # From https://codegolf.stackexchange.com/questions/4707#answer-4712
                ordinal = (lambda n: "%d%s" % (n, "tsnrhtdd"[
                           (n / 10 % 10 != 1) * (n % 10 < 4) * n % 10::4]))(o + 1)
            self.node = node
        else:
            self.node = max(self.node.children, key=lambda n: n.N)

        stats = Stats(kl_div, rolls, time.time() - start_time)
        return self.node, stats
