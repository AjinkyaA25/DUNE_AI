"""PPO policy on top of the heuristic (user, 2026-10-19: "with a NN, would it be
able to learn tech gameplay, contract playstyles ... independently? If PPO can
be useful, that should be the next step").

Every decision with 2+ legal moves is scored

    logit(a) = heuristic_score(a) / T + correction(state, a)

and a move is sampled from the softmax (greedy = argmax for evaluation). The
correction network starts at exactly 0, so the untrained policy plays like
Best AI (with exploration among near-equal moves) and PPO learns, from wins
only, where the heuristic's judgement is wrong - per situation, for every
decision type (Agent spaces, buys, techs, contracts, deploys, intrigues ...).

Inputs: the state vector (src/ai/features.encode_state: cards, decks, row,
Bloodlines techs / skills / market, playstyle), each move's dense features
(src/ai/action_features.encode_action) and the IDENTITY of what the move
touches (board space, card bought, card played, Intrigue, tech / skill /
contract) through learned embeddings - so it can learn "CHOAM Transports with
3 contracts done" or "this card at this space" without being told.

Training: src/selfplay/ppo_train.py (rollouts on CPU workers, numpy forward;
updates in PyTorch on the GPU).

spec: ppo:net=<npz>[:tuned=<json>][:greedy][:T<temp>][:K<n>]

Shortlist (2026-10-20): a net trained with --k 0 chooses among EVERY legal
move (stored as K=0 in the npz); older nets keep the heuristic's top 16.
K<n> in the spec overrides it (K0 = all).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from src.game.gameState import ActionType

K_MAX = 16                 # default moves kept per decision: the heuristic's top 16 (0 = all)
EMB = 16                   # embedding size per identity field
ID_FIELDS = ("space", "card", "played", "intrigue", "choice")
_IDS: Dict[str, Dict[str, int]] = {}


def _build_ids() -> None:
    """name -> id (0 = none) for every identity field."""
    if _IDS:
        return
    from src.data.card_definitions import setup_game
    from src.game.board.board import UPRISING_BOARD
    gs = setup_game(num_players=4, seed=0, use_bloodlines=True)
    cards = set()
    for c in list(gs.imperium_deck) + list(gs.imperium_row):
        cards.add(c.name)
    for p in gs.players:
        for z in (p.deck, p.hand, p.discard):
            for c in z:
                cards.add(c.name)
    for st in (gs.reserve_prepare_the_way, gs.reserve_spice_must_flow):
        for c in st:
            cards.add(c.name)
    cards.add("Signet Ring")
    intr = {c.name for c in gs.intrigue_deck}
    from src.game.board.board import LEADER_SPACES
    spaces = sorted(set(UPRISING_BOARD) | set(LEADER_SPACES))
    for field, names in (("space", spaces), ("card", sorted(cards)), ("played", sorted(cards)),
                         ("intrigue", sorted(intr))):
        _IDS[field] = {n: i + 1 for i, n in enumerate(names)}


CHOICE_BUCKETS = 512      # any choice string (tech, skill, contract, spy post, option ...) -> stable hash


def id_sizes() -> List[int]:
    _build_ids()
    return [len(_IDS[f]) + 2 for f in ID_FIELDS[:-1]] + [CHOICE_BUCKETS + 1]   # +1 none, +1 unknown


def _id(field: str, name) -> int:
    if not name:
        return 0
    if field == "choice":
        import zlib
        return 1 + zlib.crc32(str(name).encode()) % CHOICE_BUCKETS
    d = _IDS[field]
    return d.get(name, len(d) + 1)


def action_ids(gs, pid: int, a) -> List[int]:
    """Identity of what the move touches, one id per ID_FIELDS entry."""
    _build_ids()
    at = a.action_type
    space = a.space_name if at == ActionType.AGENT_TURN else None
    played = a.card_name if at == ActionType.AGENT_TURN else None
    card = None
    if at == ActionType.ACQUIRE_CARD:
        card = a.acquire_card_name
    elif at == ActionType.ACQUIRE_RESERVE:
        card = "The Spice Must Flow" if a.reserve_type == "spice_must_flow" else "Prepare the Way"
    intrigue = getattr(a, "intrigue_card_name", None) if at == ActionType.PLAY_INTRIGUE else None
    choice = None
    ch = getattr(a, "choice", None)
    if isinstance(ch, str) and ch:
        if ch.startswith("stack:") and getattr(gs, "bl", None) is not None:
            try:
                d, _ = gs.bl.tech_at(pid, int(ch.split(":")[1]))
                choice = f"tech:{d.name}" if d is not None else None
            except (ValueError, IndexError):
                choice = None
        elif ch.startswith(("board:", "skill:")):
            choice = "skill:" + ch.split(":", 1)[1]
        elif ch.startswith("recruit:"):
            choice = "supply" if ch.endswith("supply") else "skill:" + ch.split(":")[-1]
        else:
            choice = ch
    if at == ActionType.RESOLVE_CONTRACT:
        i = a.contract_index
        board = getattr(gs, "contracts_on_board", []) or []
        name = board[i].name if 0 <= i < len(board) else f"aside:{i}"
        choice = f"contract:{name}"
    elif at == ActionType.ACTIVATE_TECH:
        choice = f"activate:{ch}"
    return [_id("space", space), _id("card", card), _id("played", played),
            _id("intrigue", intrigue), _id("choice", choice)]


def shortlist(h: List[float], chosen: Optional[int] = None, k: int = K_MAX) -> List[int]:
    """Indices of the moves the policy chooses among: the heuristic's top k
    (k = 0: every legal move, best-first)."""
    order = sorted(range(len(h)), key=lambda i: -h[i])
    if k:
        order = order[:k]
    if chosen is not None and chosen not in order:
        order[-1] = chosen
    return order


class PPONet:
    """numpy forward of the policy / value net (weights from ppo_train.py)."""

    def __init__(self, path: Optional[str] = None, params: Optional[dict] = None):
        p = params if params is not None else dict(np.load(path))
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}
        self.T = float(self.p.get("T", np.array(1.0)))
        self.K = int(self.p.get("K", np.array(K_MAX)))       # shortlist it was trained with (0 = all)

    def state(self, s: np.ndarray):
        p = self.p
        x = np.maximum(0, p["s1.w"] @ s + p["s1.b"])
        x = np.maximum(0, p["s2.w"] @ x + p["s2.b"])
        v = float(p["v.w"] @ x + p["v.b"])
        return x, v

    def correction(self, hs: np.ndarray, D: np.ndarray, I: np.ndarray) -> np.ndarray:
        p = self.p
        e = np.concatenate([p[f"e{j}"][I[:, j]] for j in range(I.shape[1])], axis=1)
        x = np.concatenate([np.repeat(hs[None, :], D.shape[0], 0), D, e], axis=1)
        z = np.maximum(0, x @ p["a1.w"].T + p["a1.b"])
        return z @ p["a2.w"] + p["a2.b"]

    def logits(self, s, D, I, h):
        hs, v = self.state(s)
        return np.asarray(h, dtype=np.float32) / self.T + self.correction(hs, D, I), v


_NETS: Dict[str, tuple] = {}


def load_net(path: str) -> PPONet:
    """Cached by file modification time (the trainer rewrites it)."""
    mt = os.path.getmtime(path)
    hit = _NETS.get(path)
    if hit is None or hit[0] != mt:
        hit = (mt, PPONet(path))
        _NETS[path] = hit
    return hit[1]


class PPOAgent:
    name = "ppo"

    def __init__(self, net: Optional[str] = None, tuning: Optional[dict] = None, seed: Optional[int] = None,
                 greedy: bool = False, temperature: Optional[float] = None, opening_book=None,
                 k: Optional[int] = None):
        from src.ai.agents import HeuristicAgent
        self.h = HeuristicAgent(seed=seed, opening_book=opening_book, tuning=tuning)
        self.net_path = net
        self.greedy = greedy
        self.temperature = temperature
        self.k = k                               # shortlist size (None = what the net was trained with)
        self.rng = np.random.default_rng(seed)
        self.record: Optional[list] = None       # set by the trainer: decisions are appended
        self.hidden = False                      # trainer: also record what only an observer sees (critic)
        self.last: Optional[dict] = None         # the last decision's top moves (UI explanation)

    def reset(self) -> None:
        pass

    def _evaluate(self, gs, pid, valid):
        """-> heuristic scores of all moves, the shortlist, its logits, value, network inputs."""
        from src.ai.features import encode_state
        from src.ai.action_features import encode_action
        from src.ai.agents import decision_cache
        with decision_cache():
            h_all = [self.h.score(gs, pid, a) for a in valid]
        net = load_net(self.net_path)
        idx = shortlist(h_all, k=net.K if self.k is None else self.k)
        h = np.array([h_all[i] for i in idx], dtype=np.float32)
        lo, hi = float(h.min()), float(h.max())
        span = (hi - lo) or 1.0
        s = encode_state(gs, pid).astype(np.float32)
        D = np.stack([encode_action(gs, pid, valid[i], (h_all[i] - lo) / span) for i in idx]).astype(np.float32)
        I = np.array([action_ids(gs, pid, valid[i]) for i in idx], dtype=np.int64)
        T = self.temperature if self.temperature is not None else net.T
        lg, v = net.logits(s, D, I, h * (net.T / T))
        return h_all, idx, lg, v, (s, D, I, h)

    def scores(self, gs, pid, valid) -> List[float]:
        """Policy logit per legal move (moves outside the heuristic's top K: below every logit)."""
        if len(valid) == 1:
            return [0.0]
        _, idx, lg, _, _ = self._evaluate(gs, pid, valid)
        out = [float(lg.min()) - 10.0] * len(valid)
        for j, i in enumerate(idx):
            out[i] = float(lg[j])
        return out

    def select_action(self, gs, pid, valid):
        if len(valid) == 1:
            return valid[0]
        if self.net_path is None:
            return valid[int(np.argmax([self.h.score(gs, pid, a) for a in valid]))]
        h_all, idx, lg, v, (s, D, I, h) = self._evaluate(gs, pid, valid)
        z = lg - lg.max()
        pr = np.exp(z) / np.exp(z).sum()
        top = sorted(range(len(idx)), key=lambda j: -lg[j])[:6]     # for the UI's move explanation
        self.last = {"ppo": True, "cands": [valid[idx[j]] for j in top], "prob": [float(pr[j]) for j in top],
                     "heur": [float(h[j]) for j in top], "heur_top": valid[idx[0]], "value": float(v)}
        if self.greedy:
            k = int(np.argmax(lg))
            logp = 0.0
        else:
            k = int(self.rng.choice(len(pr), p=pr))
            logp = float(np.log(pr[k] + 1e-12))
        if self.record is not None:
            self.record.append({"pid": pid, "s": s.astype(np.float16), "D": D.astype(np.float16),
                                "I": I.astype(np.int16), "h": h, "k": k, "logp": logp, "v": v,
                                "round": gs.round,
                                # VP of me, next, next+1, next+2 now (dense reward)
                                "vp": [gs.players[(pid + j) % len(gs.players)].victory_points
                                       for j in range(len(gs.players))]})
            if self.hidden:                      # privileged critic input (never seen by the policy)
                from src.ai.hidden_features import encode_hidden
                self.record[-1]["hid"] = encode_hidden(gs, pid).astype(np.float16)
        return valid[idx[k]]
