"""
Structured, machine-verifiable game log.

`record_game(agents, ...)` plays one full game and returns a dict with:
  - "events": one record per non-NO_OP action, with the round / phase / player,
    the action itself (card played, space, intrigue, contract, reserve type,
    troops deployed, ...) and the exact per-player state delta it caused
  - "rounds": end-of-round standings snapshots
  - "combats": the resolved CombatResult for every round
  - "summary": per-player aggregates (cards bought, TSMF bought, Prepare the Way
    bought, intrigues played, troops deployed, contracts acquired/completed,
    spies placed, VP breakdown) plus the final result

CLI:
  python -m src.selfplay.game_log --seed 42 --players 4 \
      --agents random,heuristic,heuristic,heuristic --out game.jsonl
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from src.data.card_definitions import setup_game
from src.game.gameState import ActionType, Phase
from src.ai.agents import make_agent

_FACS = ("emperor", "spacing_guild", "bene_gesserit", "fremen")
MOVE_CAP = 5000


def _snapshot(gs, pid: int) -> Dict:
    p = gs.players[pid]
    return {
        "vp": p.victory_points,
        "solari": p.solari, "spice": p.spice, "water": p.water,
        "troops_garrison": p.troops_garrison, "troops_supply": p.troops_supply,
        "troops_in_conflict": gs.troops_in_conflict.get(pid, 0),
        "sandworms_in_conflict": gs.sandworms_in_conflict.get(pid, 0),
        "hand": len(p.hand), "deck": len(p.deck), "discard": len(p.discard),
        "in_play": len(p.in_play), "intrigue_cards": len(p.intrigue_cards),
        "persuasion_pool": gs.persuasion_pool.get(pid, 0),
        "spies_on_board": sum(p.spies_on_board.values()),
        "contracts_active": len(p.contracts_active),
        "contracts_completed": len(p.contracts_completed),
        **{f"influence_{f}": p.influence[f] for f in _FACS},
        "friendships": sum(p.faction_friendships.values()),
        "alliances": sum(p.alliances.values()),
    }


def _all_snap(gs) -> List[Dict]:
    return [_snapshot(gs, i) for i in range(gs.num_players)]


def _deltas(before: List[Dict], after: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for pid, (b, a) in enumerate(zip(before, after)):
        d = {k: a[k] - b[k] for k in b if isinstance(b[k], int) and a[k] != b[k]}
        if d:
            out[str(pid)] = d          # str keys -> stable through JSON round-trip
    return out


def _action_record(gs, a) -> Dict:
    at = a.action_type
    rec: Dict = {"action": at.value, "player": a.player_id}
    if at == ActionType.AGENT_TURN:
        rec.update(card=a.card_name, space=a.space_name,
                   space_option=a.space_option,
                   gather_intel=a.use_gather_intelligence,
                   infiltrate=a.use_infiltrate)
    elif at == ActionType.REVEAL_TURN:
        rec["revealed"] = [c.name for c in gs.players[a.player_id].hand]
    elif at == ActionType.ACQUIRE_CARD:
        rec["card"] = a.acquire_card_name
    elif at == ActionType.ACQUIRE_RESERVE:
        rec["reserve"] = a.reserve_type
    elif at == ActionType.PLAY_INTRIGUE:
        rec["intrigue"] = a.intrigue_card_name
    elif at == ActionType.RESOLVE_DEPLOY:
        rec["deploy_count"] = a.deploy_count
    elif at == ActionType.RESOLVE_TRASH:
        rec["trash_card"] = a.trash_card_name
    elif at == ActionType.RESOLVE_INFLUENCE:
        rec["faction"] = a.influence_faction
    elif at == ActionType.RESOLVE_CONTRACT:
        ct = (gs.contracts_on_board[a.contract_index]
              if a.contract_index < len(gs.contracts_on_board) else None)
        rec["contract"] = ct.name if ct else None
    elif at == ActionType.RESOLVE_SPY:
        rec["spy_post"] = a.spy_post_name
    elif at == ActionType.RESOLVE_UPLIFT:
        rec["uplift_space"] = a.uplift_space_name
    elif at == ActionType.RESOLVE_OPTIONAL:
        rec["accepted"] = a.accept_optional
    return rec


def record_game(agents, num_players: int = 4, seed: Optional[int] = None,
                leaders=None, neutral_leaders: bool = True) -> Dict:
    """`neutral_leaders` defaults to True (Leader text is unverified); pass
    `leaders=[...]` explicit names to play with real Leaders instead."""
    gs = setup_game(num_players=num_players, seed=seed, leaders=leaders,
                    neutral_leaders=neutral_leaders and leaders is None)

    events: List[Dict] = []
    rounds: List[Dict] = []
    combats: List[Dict] = []
    seen_combat = None
    last_round = 0
    move = 0

    # per-player aggregates
    agg = [dict(cards_bought=0, tsmf_bought=0, ptw_bought=0, intrigues_played=0,
                troops_deployed=0, contracts_acquired=0, spies_placed=0,
                agent_turns=0, infiltrates=0, gather_intel=0)
           for _ in range(num_players)]

    while not gs.game_over and move < MOVE_CAP:
        if gs.round != last_round:
            last_round = gs.round
            c = gs.current_conflict
            rounds.append({
                "round": gs.round, "type": "round_start",
                "conflict": c.name if c else None,
                "conflict_level": c.conflict_level if c else None,
                "battle_icon": c.battle_icon.value if c and c.battle_icon else None,
                "first_player": gs.first_player,
                "standings": _all_snap(gs),
            })

        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        action = agents[pid].select_action(gs, pid, valid)

        if action.action_type == ActionType.NO_OP:
            gs.step(action)
            move += 1
            continue

        before = _all_snap(gs)
        phase = gs.phase.value
        rec = _action_record(gs, action)
        rnd = gs.round

        gs.step(action)
        move += 1

        after = _all_snap(gs)
        rec.update(round=rnd, phase=phase, move=move,
                   leader=getattr(gs.players[action.player_id].leader, "name", "?"),
                   deltas=_deltas(before, after))
        events.append(rec)

        # aggregates
        ap = action.player_id
        at = action.action_type
        if at == ActionType.ACQUIRE_CARD:
            agg[ap]["cards_bought"] += 1
        elif at == ActionType.ACQUIRE_RESERVE:
            agg[ap]["cards_bought"] += 1
            if action.reserve_type == "spice_must_flow":
                agg[ap]["tsmf_bought"] += 1
            elif action.reserve_type == "prepare_the_way":
                agg[ap]["ptw_bought"] += 1
        elif at == ActionType.PLAY_INTRIGUE:
            agg[ap]["intrigues_played"] += 1
        elif at == ActionType.RESOLVE_DEPLOY:
            agg[ap]["troops_deployed"] += action.deploy_count
        elif at == ActionType.RESOLVE_CONTRACT:
            agg[ap]["contracts_acquired"] += 1
        elif at == ActionType.RESOLVE_SPY:
            agg[ap]["spies_placed"] += 1
        elif at == ActionType.AGENT_TURN:
            agg[ap]["agent_turns"] += 1
            agg[ap]["infiltrates"] += int(action.use_infiltrate)
            agg[ap]["gather_intel"] += int(action.use_gather_intelligence)

        res = gs._last_combat_result
        if res is not None and res is not seen_combat:
            seen_combat = res
            combats.append({
                "round": rnd, "conflict": res.conflict_name,
                "rankings": [{"player": p, "strength": s} for p, s in res.rankings],
                "has_sandworm": {int(k): v for k, v in res.has_sandworm.items() if v},
                "rewards": {int(k): v for k, v in res.rewards_awarded.items()},
                "winner": res.winner,
            })
            rounds.append({"round": rnd, "type": "round_end",
                           "standings": _all_snap(gs)})

    if not gs.game_over:
        gs.check_victory_conditions()

    final_vp = [p.victory_points for p in gs.players]
    winner = gs.winner if gs.winner is not None else int(
        max(range(num_players), key=lambda i: final_vp[i]))

    for i in range(num_players):
        agg[i].update(
            leader=getattr(gs.players[i].leader, "name", "?"),
            final_vp=final_vp[i],
            endgame_vp=getattr(gs, "_endgame_vp_gained", {}).get(i, 0),
            friendships=sum(gs.players[i].faction_friendships.values()),
            alliances=sum(gs.players[i].alliances.values()),
            controlled=sum(1 for o in gs.controlled_by.values() if o == i),
            won_conflicts=len(gs.won_conflicts.get(i, [])),
        )

    return {
        "seed": seed, "num_players": num_players,
        "leaders": [getattr(p.leader, "name", "?") for p in gs.players],
        "events": events, "rounds": rounds, "combats": combats,
        "summary": {"winner": winner, "final_vp": final_vp,
                    "rounds_played": gs.round, "moves": move,
                    "players": agg},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Record one game as structured JSONL")
    ap.add_argument("--players", type=int, default=4, choices=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agents", type=str, default="heuristic,heuristic,heuristic,heuristic")
    ap.add_argument("--out", type=str, default="game_log.jsonl")
    ap.add_argument("--real-leaders", action="store_true",
                    help="use real (unverified) Leader abilities instead of "
                         "the default no-op Leaders")
    args = ap.parse_args()

    specs = (args.agents.split(",") + ["heuristic"] * args.players)[:args.players]
    agents = {i: make_agent(specs[i] or "heuristic", seed=(args.seed or 0) * 9 + i)
              for i in range(args.players)}
    log = record_game(agents, num_players=args.players, seed=args.seed,
                      neutral_leaders=not args.real_leaders)

    with open(args.out, "w") as f:
        f.write(json.dumps({"meta": {k: log[k] for k in
                ("seed", "num_players", "leaders")}, "agents": specs}) + "\n")
        for ev in log["events"]:
            f.write(json.dumps(ev) + "\n")
        f.write(json.dumps({"final": log["summary"]}) + "\n")

    s = log["summary"]
    print(f"wrote {len(log['events'])} events -> {args.out}")
    print(f"winner P{s['winner']}  VP {s['final_vp']}  ({s['rounds_played']} rounds)")
    print(f"{'seat':>4} {'leader':22} {'agent':28} {'VP':>3} {'buys':>5} "
          f"{'TSMF':>5} {'PTW':>4} {'intr':>5} {'depl':>5} {'ctr':>4}")
    for i, pl in enumerate(s["players"]):
        print(f"{i:>4} {pl['leader']:22.22} {specs[i]:28.28} {pl['final_vp']:>3} "
              f"{pl['cards_bought']:>5} {pl['tsmf_bought']:>5} {pl['ptw_bought']:>4} "
              f"{pl['intrigues_played']:>5} {pl['troops_deployed']:>5} "
              f"{pl['contracts_acquired']:>4}")


if __name__ == "__main__":
    main()
