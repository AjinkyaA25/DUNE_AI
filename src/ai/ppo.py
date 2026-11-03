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

Leaders (user, 2026-10-23: "fine tune to each specific leader and understand
strategy for all of them"): a net with 6 embedding fields also gets the
Leader of the deciding player as an identity on every move (so each move's
correction can depend on WHO plays it), the observation post of a Spy
placement (its own field: hashing it into "choice" lands on buckets trained
for other choices) and a one-hot of every seat's Leader appended to the
state (leader_state). A Spy placement also carries plan features
(action_features.spy_plan_features: can I still go to the post's spaces this
round, could my deck later, is a rival there) when the net's move layer is
wide enough for them. Older nets (5 fields, shorter state) are read exactly
as before.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from src.game.gameState import ActionType

K_MAX = 16                 # default moves kept per decision: the heuristic's top 16 (0 = all)
EMB = 16                   # embedding size per identity field
ID_FIELDS = ("space", "card", "played", "intrigue", "choice", "leader", "post")
N_OLD_FIELDS = 5           # nets before the Leader field
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
    from src.game.leader.leader_definitions import LEADER_NAMES
    from src.game.board.board import OBSERVATION_POST_CONNECTIONS
    for field, names in (("space", spaces), ("card", sorted(cards)), ("played", sorted(cards)),
                         ("intrigue", sorted(intr)), ("leader", list(LEADER_NAMES)),
                         ("post", sorted(OBSERVATION_POST_CONNECTIONS))):
        _IDS[field] = {n: i + 1 for i, n in enumerate(names)}


CHOICE_BUCKETS = 512      # any choice string (tech, skill, contract, spy post, option ...) -> stable hash


def id_sizes() -> List[int]:
    _build_ids()
    return [len(_IDS[f]) + 2 if f != "choice" else CHOICE_BUCKETS + 1 for f in ID_FIELDS]   # +1 none, +1 unknown


def _leader_names() -> List[str]:
    from src.game.leader.leader_definitions import LEADER_NAMES
    return list(LEADER_NAMES)


LEADER_STATE_DIM = 4 * (len(_leader_names()) + 1)


def leader_state(gs, pid: int) -> np.ndarray:
    """One-hot Leader of me, next, next+1, next+2 (slot 0 = no / neutral Leader)."""
    names = _leader_names()
    n = len(names) + 1
    v = np.zeros(LEADER_STATE_DIM, np.float32)
    k = len(gs.players)
    for j in range(min(4, k)):
        name = getattr(gs.players[(pid + j) % k].leader, "name", None)
        v[j * n + (names.index(name) + 1 if name in names else 0)] = 1.0
    return v


NAV_STATE_DIM = 4 * 10 + 10 + 1      # my Navigation card per slot | cards kept (setup) | next slot / 4


def nav_state(gs, pid: int) -> np.ndarray:
    """Steersman's own face-down Navigation cards (zeros for anyone else):
    which card sits in which slot, which are kept while laying them out, and
    how far along the row he has played."""
    v = np.zeros(NAV_STATE_DIM, np.float32)
    st = getattr(gs.players[pid].leader, "state", None) or {}
    for k, c in enumerate(st.get("nav", [])[:4]):
        if 1 <= c <= 10:
            v[k * 10 + c - 1] = 1.0
    for c in st.get("nav_keep", []):
        if 1 <= c <= 10:
            v[40 + c - 1] = 1.0
    v[50] = st.get("nav_next", 0) / 4.0
    return v


def full_state(gs, pid: int, width: int) -> np.ndarray:
    """encode_state + the Leader blocks a net of this input width was built with."""
    from src.ai.features import encode_state
    s = encode_state(gs, pid).astype(np.float32)
    if width > len(s):
        s = np.concatenate([s, leader_state(gs, pid)])
    if width > len(s):
        s = np.concatenate([s, nav_state(gs, pid)])
    return s


class CriticNet:
    """numpy forward of the trainer's privileged critic (critic.npz): the value
    of a position from the policy's inputs + what only an observer knows."""

    def __init__(self, path: str):
        p = dict(np.load(path))
        self.w = [(np.asarray(p[f"w{j}"], np.float32), np.asarray(p[f"b{j}"], np.float32)) for j in range(3)]
        self.S = int(p["S"])

    def value(self, gs, pid: int) -> float:
        from src.ai.hidden_features import encode_hidden
        x = np.concatenate([full_state(gs, pid, self.S), encode_hidden(gs, pid).astype(np.float32)])
        for j, (w, b) in enumerate(self.w):
            x = w @ x + b
            if j < 2:
                x = np.maximum(0, x)
        return float(x[0])


def final_reward(gs, pid: int) -> float:
    """The trainer's game result: win (ties split) + 0.02 x VP lead (clipped to +-5)."""
    vp = [q.victory_points for q in gs.players]
    if gs.winner is not None:
        win = float(gs.winner == pid)
    else:
        tied = [i for i, v in enumerate(vp) if v == max(vp)]
        win = 1.0 / len(tied) if pid in tied else 0.0
    lead = vp[pid] - max(v for i, v in enumerate(vp) if i != pid)
    return win + 0.02 * max(-5, min(5, lead))


def critic_q(gs, pid: int, moves: list, lg: np.ndarray, critic: "CriticNet", m: int) -> np.ndarray:
    """Critic lookahead (user, 2026-10-25: "the best moves always using the
    critic"): the policy's top m moves are each played on a copy of the game
    and the position right after is valued by the all-seeing critic (a game
    that ends there: its real result). NaN = not tried."""
    q = np.full(len(moves), np.nan, np.float32)
    for j in np.argsort(-lg)[:m]:
        g2 = gs.clone()
        try:
            g2.step(moves[j])
        except Exception:
            continue
        q[j] = final_reward(g2, pid) if g2.game_over else critic.value(g2, pid)
    return q


def rollout_q(gs, pid: int, moves: list, lg: np.ndarray, critic: "CriticNet", m: int, n: int,
              net: str, rng: np.random.Generator, cap: int = 400,
              tuning: Optional[dict] = None) -> np.ndarray:
    """Rollout lookahead (2026-10-25, after the 1-step critic_q proved too noisy):
    the policy's top m moves are each played in n determinized worlds (hidden
    cards re-dealt, as ISMCTS does), every seat then plays on with the greedy
    policy to the end of the round (its Conflict resolved), and that position
    is valued by the critic (a game that ends: its real result), averaged over
    the worlds. Every move sees the same n worlds, so the comparison between
    moves isn't swamped by luck of the deal. NaN = not tried."""
    from src.ai.determinize import determinize
    q = np.full(len(moves), np.nan, np.float32)
    pol = PPOAgent(net=net, tuning=tuning if tuning is not None else _rollout_tuning(), seed=int(rng.integers(2**31)), greedy=True)
    worlds = [determinize(gs, pid, rng) for _ in range(n)]
    r0 = gs.round
    for j in np.argsort(-lg)[:m]:
        tot, cnt = 0.0, 0
        for w in worlds:
            g2 = w.clone()
            try:
                g2.step(moves[j])
                k = 0
                while not g2.game_over and g2.round == r0 and k < cap:
                    cur = g2.player_in_reveal_buy
                    if cur is None:
                        cur = g2.get_current_player_id()
                    g2.step(pol.select_action(g2, cur, g2.get_valid_actions(cur)))
                    k += 1
            except Exception:
                continue
            tot += final_reward(g2, pid) if g2.game_over else critic.value(g2, pid)
            cnt += 1
        if cnt:
            q[j] = tot / cnt
    return q


_ROLL_TUNING: Optional[dict] = None


def _rollout_tuning() -> dict:
    global _ROLL_TUNING
    if _ROLL_TUNING is None:
        import json
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "config", "ppo", "base.json"),
                  encoding="utf-8") as f:
            _ROLL_TUNING = json.load(f)
    return _ROLL_TUNING


_CRITICS: Dict[str, tuple] = {}


def load_critic(path: str) -> CriticNet:
    mt = os.path.getmtime(path)
    hit = _CRITICS.get(path)
    if hit is None or hit[0] != mt:
        hit = (mt, CriticNet(path))
        _CRITICS[path] = hit
    return hit[1]


def _id(field: str, name) -> int:
    if not name:
        return 0
    if field == "choice":
        import zlib
        return 1 + zlib.crc32(str(name).encode()) % CHOICE_BUCKETS
    d = _IDS[field]
    return d.get(name, len(d) + 1)


def action_ids(gs, pid: int, a, leader: bool = True) -> List[int]:
    """Identity of what the move touches, one id per ID_FIELDS entry
    (leader False: the first 5, as older nets were trained)."""
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
        elif ch.startswith(("nav:", "drop:")):
            # Steersman's Navigation cards: "nav:3" is the same id in every slot, so a
            # learned id said "card 3 is good" and put it in slot 1 (it pays in slot 4).
            # They go by leader_act_features (card + slot) and the heuristic instead.
            choice = None
        else:
            choice = ch
    if at == ActionType.RESOLVE_CONTRACT:
        i = a.contract_index
        board = getattr(gs, "contracts_on_board", []) or []
        name = board[i].name if 0 <= i < len(board) else f"aside:{i}"
        choice = f"contract:{name}"
    elif at == ActionType.ACTIVATE_TECH:
        choice = f"activate:{ch}"
    ids = [_id("space", space), _id("card", card), _id("played", played),
           _id("intrigue", intrigue), _id("choice", choice)]
    if leader:
        ids.append(_id("leader", getattr(gs.players[pid].leader, "name", None)))
        post = a.spy_post_name if at == ActionType.RESOLVE_SPY else None   # WHICH post (recalled Spy: heuristic)
        ids.append(_id("post", post))
    return ids


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
        self.nf = sum(1 for k in self.p if k[0] == "e" and k[1:].isdigit())   # identity fields
        self.S = self.p["s1.w"].shape[1]                     # state inputs (+ leader_state on Leader nets)
        self.leaders = self.nf > N_OLD_FIELDS
        E = self.p["e0"].shape[1] if "e0" in self.p else 0
        self.A = self.p["a1.w"].shape[1] - self.p["s2.w"].shape[0] - self.nf * E   # move features it reads

    def state(self, s: np.ndarray):
        p = self.p
        x = np.maximum(0, p["s1.w"] @ s + p["s1.b"])
        x = np.maximum(0, p["s2.w"] @ x + p["s2.b"])
        v = float(p["v.w"] @ x + p["v.b"])
        return x, v

    def correction(self, hs: np.ndarray, D: np.ndarray, I: np.ndarray) -> np.ndarray:
        p = self.p
        e = np.concatenate([p[f"e{j}"][I[:, j]] for j in range(self.nf)], axis=1)
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
        self.tuning = tuning                     # the rollout teacher's play-outs use the same heuristic
        self.greedy = greedy
        self.temperature = temperature
        self.k = k                               # shortlist size (None = what the net was trained with)
        self.rng = np.random.default_rng(seed)
        self.record: Optional[list] = None       # set by the trainer: decisions are appended
        self.hidden = False                      # trainer: also record what only an observer sees (critic)
        self.last: Optional[dict] = None         # the last decision's top moves (UI explanation)
        self.teacher: Optional[dict] = None      # trainer: {critic, frac, m, act} - grade moves by lookahead

    def reset(self) -> None:
        pass

    def _evaluate(self, gs, pid, valid):
        """-> heuristic scores of all moves, the shortlist, its logits, value, network inputs."""
        from src.ai.action_features import encode_action
        from src.ai.agents import decision_cache
        with decision_cache():
            h_all = [self.h.score(gs, pid, a) for a in valid]
        net = load_net(self.net_path)
        idx = shortlist(h_all, k=net.K if self.k is None else self.k)
        h = np.array([h_all[i] for i in idx], dtype=np.float32)
        lo, hi = float(h.min()), float(h.max())
        span = (hi - lo) or 1.0
        s = full_state(gs, pid, net.S)
        D = np.stack([encode_action(gs, pid, valid[i], (h_all[i] - lo) / span) for i in idx]).astype(np.float32)
        if net.A > D.shape[1]:                   # Spy plan features
            from src.ai.action_features import spy_plan_features
            D = np.concatenate([D, np.stack([spy_plan_features(gs, pid, valid[i]) for i in idx])], axis=1)
        if net.A > D.shape[1]:                   # Leader setup features (Steersman's Navigation slots)
            from src.ai.action_features import leader_act_features
            D = np.concatenate([D, np.stack([leader_act_features(gs, pid, valid[i]) for i in idx])], axis=1)
        I = np.array([action_ids(gs, pid, valid[i], net.leaders) for i in idx], dtype=np.int64)
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
        q, teach = None, False
        t = self.teacher
        # rollout teacher ("n" worlds): only where the policy is unsure (top move < "pmax"), it's costly
        if t is not None and self.rng.random() < t["frac"] and (not t.get("n") or pr.max() < t.get("pmax", 1.0)):
            if t.get("n"):
                q = rollout_q(gs, pid, [valid[i] for i in idx], lg, load_critic(t["critic"]), t["m"], t["n"],
                              self.net_path, self.rng, tuning=self.tuning)
            else:
                q = critic_q(gs, pid, [valid[i] for i in idx], lg, load_critic(t["critic"]), t["m"])
            # play the move the critic rates best (a tie with the policy's own pick: keep that)
            if t.get("act") and np.isfinite(q).any() and not q[k] >= np.nanmax(q) - 1e-6:
                k = int(np.nanargmax(q))
                logp = float(np.log(pr[k] + 1e-12))
                teach = True
        if self.record is not None:
            self.record.append({"pid": pid, "s": s.astype(np.float16), "D": D.astype(np.float16),
                                "I": I.astype(np.int16), "h": h, "k": k, "logp": logp, "v": v,
                                "round": gs.round,
                                # VP of me, next, next+1, next+2 now (dense reward)
                                "vp": [gs.players[(pid + j) % len(gs.players)].victory_points
                                       for j in range(len(gs.players))]})
            if q is not None:
                self.record[-1]["q"] = q.astype(np.float32)
                self.record[-1]["teach"] = teach
            if self.hidden:                      # privileged critic input (never seen by the policy)
                from src.ai.hidden_features import encode_hidden
                self.record[-1]["hid"] = encode_hidden(gs, pid).astype(np.float16)
        return valid[idx[k]]
