"""Information-set MCTS guided by the AlphaZero net.

Each simulation starts from a fresh determinization of the real game for the
searching player (src/ai/determinize.py: opponents' hands, every deck order,
the intrigue deck, ... re-dealt into one possible world), then walks the tree
with PUCT. Tree edges are keyed by the move's repr, so one tree is shared
across worlds (single-observer ISMCTS, open loop over chance). A move that
is illegal in the current world is simply not offered there. Leaves are
scored by the value head; there are no rollouts.

Four players: every node stores a value vector (one entry per seat) and each
player picks by its own entry (max^n backup).
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

from src.ai.determinize import determinize
from src.az.encode import encode_actions, encode_obs
from src.az.net import AZNet
from src.game.gameState import ActionType

C_PUCT = 1.5
# Reward shaping (src.az.train --margin / --speed; set in the environment so
# spawned workers see it): VP margin weight, and a bonus to the winner for
# each round the game ended before round 10.
MARGIN = float(os.environ.get("AZ_MARGIN", "0.02"))
SPEED = float(os.environ.get("AZ_SPEED", "0.0"))
# Scoring pace: every seat gets PACE x (its VP / rounds played), so scoring
# fast pays whatever the source - combat, TSMF, influence - and whoever wins.
PACE = float(os.environ.get("AZ_PACE", "0.0"))


def active(gs) -> int:
    p = gs.player_in_reveal_buy
    return gs.get_current_player_id() if p is None else p


def legal(gs, pid: int) -> list:
    v = gs.get_valid_actions(pid)
    return [a for a in v if a.action_type != ActionType.NO_OP] or v


def terminal_values(gs) -> np.ndarray:
    """Per seat: win share (ties split) + MARGIN x VP margin over the best
    rival, + SPEED x rounds left unplayed for the winner (faster wins),
    + PACE x VP per round played (scoring rate, every seat)."""
    if not gs.game_over:
        gs.check_victory_conditions()
    vp = [p.victory_points for p in gs.players]
    top = max(vp)
    winners = [gs.winner] if gs.winner is not None else [i for i, v in enumerate(vp) if v == top]
    out = np.zeros(len(vp), np.float32)
    for i in range(len(vp)):
        win = 1.0 / len(winners) if i in winners else 0.0
        out[i] = win + MARGIN * (vp[i] - max(v for j, v in enumerate(vp) if j != i))
        if win and SPEED:
            out[i] += win * SPEED * max(0, 10 - gs.round)
        if PACE:
            out[i] += PACE * vp[i] / max(1, gs.round)
    return out


def evaluate(net: AZNet, gs, pid: int, acts) -> tuple:
    """(prior probabilities over acts, value vector by absolute seat)."""
    h = net.trunk(encode_obs(gs, pid))
    dense, idx = encode_actions(gs, pid, acts)
    lg = net.logits(h, dense, idx)
    lg = lg - lg.max()
    pr = np.exp(lg)
    pr /= pr.sum()
    rel = net.value(h)
    n = gs.num_players
    v = np.zeros(n, np.float32)
    for k in range(n):
        v[(pid + k) % n] = rel[k]
    return pr, v


class Node:
    __slots__ = ("N", "W", "P", "children")

    def __init__(self, n_players: int):
        self.N = 0
        self.W = np.zeros(n_players, np.float32)
        self.P: Optional[Dict[str, float]] = None
        self.children: Dict[str, "Node"] = {}


def _select(node: Node, cur: int, acts, keys, net, g, n_players):
    missing = [k for k in keys if k not in node.P]
    if missing:                              # a move only legal in this world
        pr, _ = evaluate(net, g, cur, acts)
        for k, p in zip(keys, pr):
            node.P.setdefault(k, float(p))
    tot = sum(node.P[k] for k in keys) or 1.0
    sq = math.sqrt(node.N + 1)
    fpu = (node.W[cur] / node.N - 0.1) if node.N else 0.25
    best, bi = -1e9, 0
    for i, k in enumerate(keys):
        ch = node.children.get(k)
        n = ch.N if ch else 0
        q = ch.W[cur] / n if n else fpu
        s = q + C_PUCT * (node.P[k] / tot) * sq / (1 + n)
        if s > best:
            best, bi = s, i
    k = keys[bi]
    ch = node.children.get(k)
    if ch is None:
        ch = node.children[k] = Node(n_players)
    return acts[bi], ch


def search(net: AZNet, gs, pid: int, acts, sims: int, rng: np.random.Generator,
           noise: float = 0.25, alpha: float = 0.3, max_depth: int = 400) -> np.ndarray:
    """Visit counts over `acts` (the real game's legal moves for pid)."""
    n_players = gs.num_players
    root = Node(n_players)
    keys0 = [repr(a) for a in acts]
    pr, _ = evaluate(net, gs, pid, acts)
    if noise:
        pr = (1 - noise) * pr + noise * rng.dirichlet([alpha] * len(acts))
    root.P = dict(zip(keys0, map(float, pr)))
    for _ in range(sims):
        g = determinize(gs, pid, rng)
        node, path, depth = root, [root], 0
        v = None
        while True:
            if g.game_over:
                v = terminal_values(g)
                break
            cur = active(g)
            la = legal(g, cur)
            if len(la) == 1:                     # forced: no decision node
                g.step(la[0])
                depth += 1
                if depth > max_depth:
                    break
                continue
            keys = [repr(a) for a in la]
            if node.P is None:                   # leaf: expand with the net
                pr, v = evaluate(net, g, cur, la)
                node.P = dict(zip(keys, map(float, pr)))
                break
            a, node = _select(node, cur, la, keys, net, g, n_players)
            path.append(node)
            g.step(a)
            depth += 1
            if depth > max_depth:
                break
        if v is None:
            continue
        for nd in path:
            nd.N += 1
            nd.W += v
    return np.array([root.children[k].N if k in root.children else 0 for k in keys0],
                    np.float32)
