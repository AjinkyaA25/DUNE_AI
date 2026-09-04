"""
Agents for Dune Imperium: Uprising.

  RandomAgent       - uniform over non-NO_OP legal actions
  HeuristicAgent    - fast hand-crafted scoring of each legal action (no search)
  GreedyValueAgent  - 1-ply lookahead: clone, apply, evaluate with a ValueModel
                      (falls back to the heuristic state value if no model)

All agents expose `select_action(gs, pid, valid_actions) -> GameAction`.
`temperature > 0` turns the argmax into a softmax sample (for self-play).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np

from src.game.gameState import GameState, GameAction, ActionType
from src.ai.features import encode_state
from src.ai.value_model import ValueModel
from src.ai.opening_book import OpeningBook

# rough marginal value of one unit of each resource (VP-equivalent * 10)
_RES_VALUE = {
    "vp": 10.0, "solari": 0.55, "spice": 0.65, "water": 0.55,
    "troops": 0.8, "draw": 1.1, "intrigue": 1.3, "persuasion": 0.8,
    "maker_hooks": 1.5, "uplift": 3.0, "spy": 1.2, "spy_special": 1.4,
    "influence_emperor": 1.6, "influence_spacing_guild": 1.6,
    "influence_bene_gesserit": 1.6, "influence_fremen": 1.6,
    "sandworm": 3.0, "sandworm_maker_space": 3.0,
}
_INF_KEYS = ("influence_emperor", "influence_spacing_guild",
             "influence_bene_gesserit", "influence_fremen")
_FACTION_OF = {
    "influence_emperor": "emperor", "influence_spacing_guild": "spacing_guild",
    "influence_bene_gesserit": "bene_gesserit", "influence_fremen": "fremen",
}


# ---------------------------------------------------------------------------

def _effect_value(gs: GameState, pid: int, eff: dict) -> float:
    p = gs.players[pid]
    v = 0.0
    for k, amt in eff.items():
        if not isinstance(amt, (int, float)):
            continue
        base = _RES_VALUE.get(k, 0.3)
        v += base * amt
        if k in _FACTION_OF:                         # threshold bonuses
            cur = p.influence[_FACTION_OF[k]]
            if cur < 2 <= cur + amt:
                v += 6.0                             # friendship = 1 VP
            if cur < 4 <= cur + amt:
                v += 5.0                             # alliance push
    return v


def _flatten(eff: dict) -> dict:
    """Best-effort flatten of nested conditional/choice sub-effects for scoring."""
    out = {}
    for k, v in eff.items():
        if isinstance(v, dict) and (k.startswith("if_") or k in (
                "pay_then", "recall_spy_then", "discard_then", "choose_by_combat")):
            inner = v.get("combat") or v.get("else") or v
            for kk, vv in (inner.items() if isinstance(inner, dict) else []):
                if isinstance(vv, (int, float)):
                    out[kk] = out.get(kk, 0) + vv
        elif isinstance(v, (int, float)):
            out[k] = out.get(k, 0) + v
    return out


def _cond_weight(gs: GameState, pid: int, key: str) -> float:
    """
    0..1 'how likely/true is this condition for me right now' — used to value
    a card's CONDITIONAL agent/reveal effects at buy time, so a card whose
    only payoff sits behind an alliance/influence threshold the player doesn't
    have isn't priced the same as an unconditional card of the same cost
    (e.g. Junction Headquarters without the Spacing Guild Alliance).
    """
    p = gs.players[pid]
    if key.startswith("alliance_"):
        return 1.0 if p.alliances.get(key[9:]) else 0.10
    if key == "any_alliance":
        return 1.0 if any(p.alliances.values()) else 0.10
    if key.startswith("influence_"):                # influence_fremen_2
        fac, _, n = key[10:].rpartition("_")
        need = int(n) if n.isdigit() else 2
        cur = p.influence.get(fac, 0)
        if cur >= need:
            return 1.0
        return max(0.10, 0.40 - 0.10 * (need - cur))
    if key.startswith("tag_other_"):
        return 0.30                                  # rarely true yet, some upside
    if key.startswith("contracts_"):
        need = int(key[10:]) if key[10:].isdigit() else 2
        return 1.0 if len(p.contracts_completed) >= need else 0.15
    if key.startswith("spies_"):
        need = int(key[6:]) if key[6:].isdigit() else 2
        return 1.0 if sum(p.spies_on_board.values()) >= need else 0.30
    if key.startswith("units_in_conflict_"):
        return 0.30                                  # depends on future combat state
    if key == "fremen_bond":
        return 1.0 if p.influence.get("fremen", 0) >= 2 else 0.30
    if key == "councilor":
        return 1.0 if p.has_councilor else 0.15
    if key == "swordmaster":
        return 1.0 if p.has_swordmaster else 0.15
    return 0.35                                      # unknown / per-turn flag: modest default


def _card_effect_value(gs: GameState, pid: int, eff: dict, w: float = 1.0) -> float:
    """
    Recursively value a card's agent/reveal effect dict, discounting
    conditional sub-effects by how likely their condition is to hold
    (see `_cond_weight`) instead of either counting them at full value or
    silently ignoring them.
    """
    v = 0.0
    for k, sub in eff.items():
        if k.startswith("if_") and isinstance(sub, dict):
            v += _card_effect_value(gs, pid, sub, w * _cond_weight(gs, pid, k[3:]))
        elif k == "pay_then" and isinstance(sub, dict):
            cost = sub.get("cost", {})
            p = gs.players[pid]
            affordable = all(getattr(p, ck, 0) >= cv for ck, cv in cost.items())
            reward = {kk: vv for kk, vv in sub.items() if kk != "cost"}
            v += _card_effect_value(gs, pid, reward, w * (0.85 if affordable else 0.30))
        elif k == "choose_by_combat" and isinstance(sub, dict):
            best = max(_card_effect_value(gs, pid, sub.get("combat", {}), w),
                       _card_effect_value(gs, pid, sub.get("else", {}), w))
            v += 0.6 * best                          # situational — partial credit
        elif k in ("discard_then", "discard_then_sg", "discard_then_if_sg",
                  "recall_spy_then") and isinstance(sub, dict):
            inner = sub.get("base", sub)
            v += 0.7 * _card_effect_value(gs, pid, inner, w)
        elif k == "trash_intrigue_for" and isinstance(sub, dict):
            has_intrigue = bool(gs.players[pid].intrigue_cards)
            v += _card_effect_value(gs, pid, sub, w * (0.7 if has_intrigue else 0.25))
        elif isinstance(sub, (int, float)):
            v += w * _RES_VALUE.get(k, 0.3) * sub
            if k in _FACTION_OF:
                cur = gs.players[pid].influence[_FACTION_OF[k]]
                if cur < 2 <= cur + sub:
                    v += w * 6.0
                if cur < 4 <= cur + sub:
                    v += w * 5.0
        # other nested dicts (unrecognized structure) contribute nothing,
        # same as before — but recognized ones are no longer invisible.
    return v


def _intrigue_value(gs: GameState, pid: int, ic) -> float:
    """How good is it to play this Intrigue right now?"""
    p = gs.players[pid]
    in_combat = gs.phase.name == "COMBAT"
    from src.game.intrigue.intrigue import IntrigueTiming
    ts = ic.timing if isinstance(ic.timing, (set, tuple, list, frozenset)) else (ic.timing,)
    v = 0.0
    for e in ic.effects:
        fe = _flatten(e)
        v += _effect_value(gs, pid, fe)
        v += 0.9 * fe.get("swords", 0)          # swords matter in combat
    if IntrigueTiming.COMBAT in ts:
        if in_combat:
            worth = _conflict_worth(gs, pid)
            mine = gs.combat_strength.get(pid, 0)
            opp = max((gs.combat_strength.get(q, 0)
                       for q in range(gs.num_players) if q != pid), default=0)
            # most valuable when a swing would flip the placing
            v += 1.5 * worth if abs(mine - opp) <= 5 else 0.3 * worth
        else:
            v -= 3.0                            # save combat intrigues for combat
    if IntrigueTiming.ENDGAME in ts and not gs.game_over:
        v -= 5.0                                # never waste an endgame card early
    if list(ts) == [IntrigueTiming.PLOT] and not in_combat:
        v -= 0.6                                # small bias to hold plot intrigues
    return v


def _conflict_worth(gs: GameState, pid: int) -> float:
    """How valuable is winning the current Conflict for this player (~0..4)."""
    cc = gs.current_conflict
    if cc is None:
        return 0.0
    r = cc.first_place_reward
    v = {1: 0.5, 2: 1.8, 3: 2.8}.get(cc.conflict_level, 1.0)
    v += 3.0 * r.get("vp", 0)
    if r.get("control"):
        v += 1.6
    if any(k.startswith("may_pay") for k in r):
        v += 1.2
    v += sum(_RES_VALUE.get(k, 0.3) * n for k, n in r.items()
             if isinstance(n, (int, float)))
    # a controlled-location Conflict the player already holds is worth defending
    if cc.location and gs.controlled_by.get(cc.location) == pid:
        v += 1.0
    return v


def heuristic_state_value(gs: GameState, pid: int) -> float:
    """Cheap position eval in ~[0,1] — P(this player is doing well)."""
    p = gs.players[pid]
    others = [q for q in gs.players if q.id != pid]
    my = p.victory_points
    best_opp = max((q.victory_points for q in others), default=0)
    score = 1.5 * (my - best_opp)
    score += 0.10 * (p.solari + p.spice) + 0.08 * p.water
    score += 0.25 * p.troops_garrison + 0.5 * gs.troops_in_conflict.get(pid, 0)
    score += 0.15 * (len(p.deck) + len(p.discard) + len(p.in_play))
    score += 0.9 * sum(min(i, 4) for i in p.influence.values())
    score += 1.2 * p.agents_total + 1.5 * (1 if p.has_councilor else 0)
    score += 1.0 * len(p.intrigue_cards) + 3.0 * len(gs.won_conflicts.get(pid, []))
    return 1.0 / (1.0 + math.exp(-0.06 * score))


# ---------------------------------------------------------------------------

class Agent:
    name = "agent"

    def reset(self) -> None:
        pass

    def select_action(self, gs: GameState, pid: int,
                      valid_actions: List[GameAction]) -> GameAction:
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_action(self, gs, pid, valid_actions):
        pool = [a for a in valid_actions if a.action_type != ActionType.NO_OP]
        return self.rng.choice(pool or valid_actions)


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, seed: Optional[int] = None,
                 opening_book: Optional[OpeningBook] = None,
                 temperature: float = 0.0):
        self.rng = random.Random(seed)
        self.book = opening_book if opening_book is not None else OpeningBook.default()
        self.temperature = temperature

    # -- per-action score ----------------------------------------------

    def score(self, gs: GameState, pid: int, a: GameAction) -> float:
        p = gs.players[pid]
        at = a.action_type
        s = 0.0

        if at == ActionType.AGENT_TURN:
            from src.game.board.board import UPRISING_BOARD, SPACE_MANDATORY_COSTS
            sp = UPRISING_BOARD[a.space_name]
            for eff in gs.get_space_effects_preview(a.space_name):
                s += _effect_value(gs, pid, eff)
            fac = gs._faction_for_space(a.space_name)
            if fac:
                cur = p.influence[fac]
                s += 1.4 + (6.0 if cur == 1 else 0.0) + (4.0 if cur == 3 else 0.0)
            cost = SPACE_MANDATORY_COSTS.get(a.space_name, {})
            s -= sum(_RES_VALUE.get(k, 0.4) * v for k, v in cost.items())
            if a.space_name == "High Council" and not p.has_councilor:
                s += 5.0
            if a.space_name == "Swordmaster":
                s += 7.0
            if sp.is_combat_space and gs.current_conflict is not None:
                s += _conflict_worth(gs, pid) * (1.0 + 0.3 * min(p.troops_garrison, 4))
            if a.space_option == "pay_spice" and p.spice < 3:
                s -= 1.0
            if a.use_gather_intelligence:
                s += 0.8
            s += 4.0 * self.book.bonus(gs, pid, a)
            # discourage wasting the last agent on a weak play
            s += 0.5

        elif at == ActionType.REVEAL_TURN:
            # revealing is fine once agent plays are weak; slight positive base
            s = 1.0 + 0.4 * len(p.hand)

        elif at == ActionType.ACQUIRE_CARD:
            card = next((c for c in gs.imperium_row if c.name == a.acquire_card_name), None)
            if card:
                s += 1.5 * card.persuasion + 0.9 * card.swords
                for e in (getattr(card, "agent_effects", []) +
                          getattr(card, "reveal_effects", [])):
                    s += 0.8 * _card_effect_value(gs, pid, e)
                # NOTE: no flat "expensive = good" bonus — a costly card whose
                # payoff sits behind a condition you don't meet (e.g. Junction
                # Headquarters without the Spacing Guild Alliance) is priced by
                # what it actually does for you right now, not its cost.
                s -= 0.15 * card.cost
                s += 5.0 * self.book.bonus(gs, pid, a)

        elif at == ActionType.ACQUIRE_RESERVE:
            s = 3.0 if a.reserve_type == "spice_must_flow" else 1.2

        elif at == ActionType.PLAY_INTRIGUE:
            ic = next((c for c in p.intrigue_cards
                       if c.name == a.intrigue_card_name), None)
            if ic:
                s += _intrigue_value(gs, pid, ic)

        elif at == ActionType.END_REVEAL:
            s = -0.5 + (2.0 if gs.persuasion_pool.get(pid, 0) < 2 else -1.0)

        elif at == ActionType.COMBAT_PASS:
            s = 0.0

        elif at == ActionType.RESOLVE_DEPLOY:
            worth = _conflict_worth(gs, pid)          # 0..~4
            n = a.deploy_count
            my = gs.troops_in_conflict.get(pid, 0) + n
            opp = max((gs.troops_in_conflict.get(q, 0)
                       for q in range(gs.num_players) if q != pid), default=0)
            # value each committed troop by the reward at stake; bonus for
            # actually pulling ahead of / catching the leader, penalty for
            # over-committing when already far ahead or hopelessly behind
            s = n * (0.15 + 0.7 * worth)
            if worth > 0.5:
                if my > opp:
                    s += 1.2
                if 0 < opp - my <= 2:
                    s += 0.8 * worth
                if my - opp > 3:
                    s -= 0.5 * (my - opp - 3)
            s -= 0.35 * n                              # troops have holding value
            s -= 0.20 * n * max(0, 3 - gs.round)       # early troops are precious
            if p.troops_garrison - n < 1 and worth < 2.0:
                s -= 1.5                                # don't empty the garrison cheaply

        elif at == ActionType.RESOLVE_TRASH:
            # thin the weak starter cards; keep bought cards
            if a.trash_card_name in ("Reconnaissance", "Diplomacy",
                                     "Dune, the Desert Planet", "Dagger"):
                s = 2.0
            elif a.trash_card_name is None:
                s = 0.5
            else:
                s = -1.0

        elif at == ActionType.RESOLVE_INFLUENCE:
            cur = p.influence[a.influence_faction]
            s = 1.0 + (5.0 if cur == 1 else 0.0) + (3.0 if cur == 3 else 0.0)

        elif at == ActionType.RESOLVE_CONTRACT:
            ct = (gs.contracts_on_board[a.contract_index]
                  if a.contract_index < len(gs.contracts_on_board) else None)
            s = 1.0
            if ct is not None:
                s += sum(_effect_value(gs, pid, {k: v}) for k, v in ct.rewards.items()
                         if isinstance(v, (int, float)))
                if getattr(ct, "contract_type", None) is not None and \
                        ct.contract_type.value == "immediate":
                    s += 1.5

        elif at == ActionType.RESOLVE_SPY:
            s = 0.5
        elif at == ActionType.RESOLVE_UPLIFT:
            s = 2.0
        elif at == ActionType.RESOLVE_INTRIGUE_TRASH:
            s = 1.0
        elif at == ActionType.RESOLVE_OPTIONAL:
            op = next((o for o in gs.pending_optional_payments
                       if o.player_id == pid), None)
            if op is None:
                s = 0.0
            elif not a.accept_optional:
                s = 0.2                                   # declining is free
            else:
                gain = _effect_value(gs, pid, op.reward)
                cost = sum(_RES_VALUE.get(k, 0.3) * v for k, v in op.cost.items())
                cost += 1.1 * op.discard                  # a discarded card ~ 1 draw
                s = gain - cost
        elif at == ActionType.NO_OP:
            s = -5.0
        return s

    def select_action(self, gs, pid, valid_actions):
        scores = [self.score(gs, pid, a) for a in valid_actions]
        if self.temperature <= 0:
            best = max(range(len(valid_actions)), key=lambda i: scores[i])
            return valid_actions[best]
        return _softmax_pick(valid_actions, scores, self.temperature, self.rng)


class GreedyValueAgent(Agent):
    name = "value"

    def __init__(self, model: Optional[ValueModel] = None,
                 rollout: Optional[HeuristicAgent] = None,
                 branch_cap: int = 14, temperature: float = 0.0,
                 seed: Optional[int] = None,
                 opening_book: Optional[OpeningBook] = None,
                 heuristic_weight: float = 0.35):
        self.model = model
        self.rollout = rollout or HeuristicAgent(seed=seed, opening_book=opening_book)
        self.branch_cap = branch_cap
        self.temperature = temperature
        self.heuristic_weight = heuristic_weight   # anchor lookahead to the heuristic prior
        self.rng = random.Random(seed)
        self.book = self.rollout.book

    def _leaf(self, gs: GameState, pid: int) -> float:
        if gs.game_over:
            return 1.0 if gs.winner == pid else 0.0
        if self.model is not None:
            return self.model.predict(encode_state(gs, pid))
        return heuristic_state_value(gs, pid)

    def select_action(self, gs, pid, valid_actions):
        if self.model is None:
            # No value network yet -> just be the (well-tuned) heuristic.
            return self.rollout.select_action(gs, pid, valid_actions)
        acts = [a for a in valid_actions if a.action_type != ActionType.NO_OP] \
            or valid_actions
        if len(acts) == 1:
            return acts[0]
        # heuristic prior over every candidate (also used to prune)
        hs = [self.rollout.score(gs, pid, a) for a in acts]
        if len(acts) > self.branch_cap:
            keep = sorted(range(len(acts)), key=lambda i: hs[i],
                          reverse=True)[: self.branch_cap]
            acts = [acts[i] for i in keep]
            hs = [hs[i] for i in keep]
        lo, hi = min(hs), max(hs)
        span = (hi - lo) or 1.0
        vals = []
        for a, h in zip(acts, hs):
            g2 = gs.clone()
            try:
                g2.step(a)
                leaf = self._leaf(g2, pid)
            except Exception:
                leaf = -1.0
            h_norm = (h - lo) / span                     # 0..1
            vals.append(leaf
                        + self.heuristic_weight * h_norm
                        + 0.03 * self.book.bonus(gs, pid, a))
        if self.temperature <= 0:
            return acts[max(range(len(acts)), key=lambda i: vals[i])]
        return _softmax_pick(acts, vals, self.temperature, self.rng)


def _softmax_pick(actions, scores, temp, rng):
    m = max(scores)
    exps = [math.exp((s - m) / max(temp, 1e-6)) for s in scores]
    tot = sum(exps)
    r = rng.random() * tot
    acc = 0.0
    for a, e in zip(actions, exps):
        acc += e
        if r <= acc:
            return a
    return actions[-1]


# ---------------------------------------------------------------------------

def make_agent(spec: str, seed: Optional[int] = None,
               opening_book: Optional[OpeningBook] = None) -> Agent:
    """
    spec: 'random' | 'heuristic' | 'heuristic:T<temp>' |
          'value' | 'value:<model.npz>' | 'value:<model.npz>:T<temp>'
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        temp = 0.0
        for p in parts[1:]:
            if p.startswith("T"):
                temp = float(p[1:])
        return HeuristicAgent(seed=seed, opening_book=opening_book, temperature=temp)
    if kind == "value":
        model, temp = None, 0.0
        for p in parts[1:]:
            if p.startswith("T"):
                temp = float(p[1:])
            elif p:
                model = ValueModel.load(p)
        return GreedyValueAgent(model=model, temperature=temp, seed=seed,
                                opening_book=opening_book)
    raise ValueError(f"Unknown agent spec: {spec!r}")
