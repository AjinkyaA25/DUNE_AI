#!/usr/bin/env python
"""
Simulate one COMPLETE game of Dune Imperium: Uprising, narrated phase-by-phase
following the rulebook's 5-phase round structure:

  Phase 1  Round Start   - reveal Conflict card, draw 5 cards each
  Phase 2  Player Turns  - Agent turns / Reveal turns in turn order
  Phase 3  Combat        - strengths, Combat Intrigue, resolution & rewards
  Phase 4  Makers        - bonus spice on unoccupied desert spaces
  Phase 5  Recall        - recall Agents, pass First Player marker
  (end of round) check Victory: 10+ VP or empty Conflict deck

Usage:
  python simulate_game.py                 # 4 players, heuristic bots, random seed
  python simulate_game.py --seed 42 --players 4
  python simulate_game.py --agents value:models/value_best.npz,heuristic,heuristic,heuristic
"""
from __future__ import annotations

import argparse

from src.data.card_definitions import setup_game
from src.game.gameState import ActionType, Phase
from src.ai.agents import make_agent

BAR = "=" * 78
SUB = "-" * 78
FAC = {"emperor": "Emperor", "spacing_guild": "Spacing Guild",
       "bene_gesserit": "Bene Gesserit", "fremen": "Fremen"}


def _res(p):
    return f"sol {p.solari}  spice {p.spice}  water {p.water}"


def _inf(p):
    return "  ".join(f"{FAC[f][:3]} {p.influence[f]}" for f in FAC)


def _reward_str(d):
    if not d:
        return "-"
    parts = []
    for k, v in d.items():
        if k == "control":
            parts.append("CONTROL")
        elif k == "influence_any":
            parts.append(f"{v} influence (choice)")
        elif k.startswith("may_pay"):
            parts.append(f"may pay {v['cost']} {k.split('_')[2]} -> {v['vp']} VP")
        elif k == "trash":
            parts.append("trash a card")
        elif k == "intrigue":
            parts.append(f"{v} intrigue")
        else:
            parts.append(f"{v} {k}")
    return ", ".join(parts)


def action_desc(gs, a):
    p = gs.players[a.player_id]
    if a.action_type == ActionType.AGENT_TURN:
        mods = []
        if a.space_option:
            mods.append(a.space_option)
        if a.use_gather_intelligence:
            mods.append("Gather Intelligence")
        if a.use_infiltrate:
            mods.append("Infiltrate")
        extra = f" ({', '.join(mods)})" if mods else ""
        return f"plays '{a.card_name}', sends an Agent to {a.space_name}{extra}"
    if a.action_type == ActionType.REVEAL_TURN:
        hand = ", ".join(c.name for c in p.hand) or "(empty)"
        return f"takes a REVEAL turn - reveals: {hand}"
    if a.action_type == ActionType.ACQUIRE_CARD:
        return f"acquires '{a.acquire_card_name}' from the Imperium Row"
    if a.action_type == ActionType.ACQUIRE_RESERVE:
        return f"acquires reserve card ({a.reserve_type})"
    if a.action_type == ActionType.PLAY_INTRIGUE:
        return f"plays Intrigue '{a.intrigue_card_name}'"
    if a.action_type == ActionType.RESOLVE_DEPLOY:
        return (f"deploys {a.deploy_count} troop(s) to the Conflict"
                if a.deploy_count else "deploys no troops")
    if a.action_type == ActionType.RESOLVE_TRASH:
        return (f"trashes '{a.trash_card_name}'" if a.trash_card_name
                else "declines to trash")
    if a.action_type == ActionType.RESOLVE_INFLUENCE:
        return f"gains 1 Influence with {FAC[a.influence_faction]}"
    if a.action_type == ActionType.RESOLVE_CONTRACT:
        ct = (gs.contracts_on_board[a.contract_index]
              if a.contract_index < len(gs.contracts_on_board) else None)
        return f"takes the contract '{ct.name}'" if ct else "takes 2 Solari (no contracts left)"
    if a.action_type == ActionType.RESOLVE_SPY:
        return f"places a Spy at {a.spy_post_name}"
    if a.action_type == ActionType.RESOLVE_UPLIFT:
        return f"recalls an Agent from {a.uplift_space_name}"
    if a.action_type == ActionType.END_REVEAL:
        return "finishes buying"
    if a.action_type == ActionType.COMBAT_PASS:
        return "passes (Combat)"
    return a.action_type.value


def print_conflict(gs):
    c = gs.current_conflict
    if not c:
        return
    print(f"\n  CONFLICT revealed:  {c.name}   (Level {c.conflict_level}"
          f"{', ' + c.location if c.location else ''}"
          f"{', battle icon: ' + c.battle_icon.value if c.battle_icon else ''})")
    print(f"     1st: {_reward_str(c.first_place_reward)}")
    print(f"     2nd: {_reward_str(c.second_place_reward)}")
    if c.third_place_reward:
        print(f"     3rd: {_reward_str(c.third_place_reward)}")


def print_standings(gs, label="STANDINGS"):
    print(f"\n  {label}:")
    for p in sorted(gs.players, key=lambda x: -x.victory_points):
        al = [FAC[f] for f in FAC if p.alliances[f]]
        fr = sum(p.faction_friendships.values())
        print(f"     P{p.id} {p.leader.name:22s}  {p.victory_points:2d} VP   "
              f"{_res(p)}   friendships {fr}"
              + (f"   alliances: {', '.join(al)}" if al else ""))


def _print_combat_and_round_end(gs, res, strengths, vp_pre, rnd):
    print("\n  --- PHASE 3: COMBAT ---")
    if not res.rankings:
        print(f"     {res.conflict_name}: no player had units in the Conflict.")
    else:
        print(f"     {res.conflict_name}:")
        for place, (pid, st) in enumerate(res.rankings, 1):
            sw = "  +sandworm (rewards doubled)" if res.has_sandworm.get(pid) else ""
            print(f"        #{place}  P{pid}  strength {st}{sw}")
        for pid in gs.turn_order:
            rw = res.rewards_awarded.get(pid)
            if rw:
                gained = (gs.players[pid].victory_points - vp_pre[pid]) if vp_pre else 0
                extra = f"   (+{gained} VP total)" if gained else ""
                print(f"        P{pid} reward: {_reward_str(rw)}{extra}")
        if res.winner is not None:
            print(f"        P{res.winner} wins the Conflict card"
                  + (f" (battle icon: {gs.won_conflicts[res.winner][-1].battle_icon.value})"
                     if gs.won_conflicts[res.winner] and
                     gs.won_conflicts[res.winner][-1].battle_icon else ""))
    ctl = {loc: o for loc, o in gs.controlled_by.items() if o is not None}
    if ctl:
        print(f"     Controlled locations: "
              + ", ".join(f"{loc}=P{o}" for loc, o in ctl.items()))

    if any(q.victory_points >= 10 for q in gs.players):
        leaders = [f"P{q.id} ({q.victory_points} VP)" for q in gs.players
                   if q.victory_points >= 10]
        print(f"\n  >>> {', '.join(leaders)} at 10+ VP - the Endgame is triggered, "
              "but this round still finishes (Makers + Recall) before the game ends.")

    print("\n  --- PHASE 4: MAKERS ---")
    mb = {s: v for s, v in gs.maker_bonus_spice.items() if v}
    print(f"     +1 bonus spice on each unoccupied desert space. Accumulated: {mb or 'none'}")

    print("\n  --- PHASE 5: RECALL ---")
    print(f"     All Agents recalled to Leaders. First Player marker -> P{gs.first_player}.")
    print_standings(gs, f"END OF ROUND {rnd} - STANDINGS")


def run(num_players, seed, agent_specs, neutral_leaders=True):
    gs = setup_game(num_players=num_players, seed=seed,
                    neutral_leaders=neutral_leaders)
    agents = {i: make_agent((agent_specs[i] if agent_specs and i < len(agent_specs)
                             and agent_specs[i] else "heuristic"),
                            seed=(seed or 0) * 9 + i)
              for i in range(num_players)}

    print(BAR)
    print(f"  DUNE IMPERIUM: UPRISING  -  full game simulation  (seed {seed}, "
          f"{num_players} players)")
    print(BAR)
    for p in gs.players:
        print(f"  P{p.id}: Leader = {p.leader.name}")
    print(f"  First Player marker: P{gs.first_player}")
    print(f"  Conflict deck (this game): "
          + " -> ".join(f"L{c.conflict_level}" for c in [gs.current_conflict] + gs.conflict_deck))

    cur_round = 0
    last_combat = None
    combat_strengths = {}
    vp_pre_combat = None
    move = 0

    def open_round(r):
        print(f"\n{BAR}\n  ROUND {r}\n{BAR}")
        print("\n  --- PHASE 1: ROUND START ---")
        print_conflict(gs)
        print("  Each player draws 5 cards (Phase 1).")
        print("\n  --- PHASE 2: PLAYER TURNS ---")

    while not gs.game_over and move < 4000:
        if gs.round != cur_round:
            cur_round = gs.round
            open_round(cur_round)

        pid = gs.player_in_reveal_buy
        if pid is None:
            pid = gs.get_current_player_id()
        valid = gs.get_valid_actions(pid)
        a = agents[pid].select_action(gs, pid, valid)
        p = gs.players[pid]
        _snap = lambda: {
            "VP": p.victory_points, "sol": p.solari, "spice": p.spice,
            "wat": p.water, "trp": p.troops_garrison,
            "cards": len(p.hand) + len(p.deck) + len(p.discard) + len(p.in_play),
            "intr": len(p.intrigue_cards), "pers": gs.persuasion_pool.get(pid, 0),
            "cnf": gs.troops_in_conflict.get(pid, 0) + gs.sandworms_in_conflict.get(pid, 0),
            "ctr": len(p.contracts_active), "done": len(p.contracts_completed),
            "spy": sum(p.spies_on_board.values())}
        before = _snap()
        binf = dict(p.influence)
        prev_phase = gs.phase
        desc = action_desc(gs, a)
        if prev_phase == Phase.COMBAT:
            combat_strengths = dict(gs.combat_strength)
            vp_pre_combat = [q.victory_points for q in gs.players]

        gs.step(a)
        move += 1

        if a.action_type != ActionType.NO_OP:
            after = _snap()
            delta = [f"{k}{after[k]-before[k]:+d}" for k in before if after[k] != before[k]]
            for f in FAC:
                if p.influence[f] != binf[f]:
                    delta.append(f"{FAC[f][:3]}inf{p.influence[f]-binf[f]:+d}")
            dstr = (f"   [{' '.join(delta)}]"
                    if delta and a.action_type != ActionType.COMBAT_PASS else "")
            tag = f"COMBAT P{pid}" if prev_phase == Phase.COMBAT else f"P{pid}"
            print(f"     {tag}: {desc}{dstr}")

        # A combat just resolved (new CombatResult object)?
        res = gs._last_combat_result
        if res is not None and res is not last_combat:
            last_combat = res
            _print_combat_and_round_end(gs, res, combat_strengths, vp_pre_combat,
                                        cur_round)

    print(f"\n{BAR}\n  GAME OVER  (after round {gs.round}, "
          + ("a player reached 10 VP" if any(p.victory_points >= 10 for p in gs.players)
             else "10 rounds played / Conflict deck empty") + ")")
    print(BAR)
    eg = getattr(gs, "_endgame_vp_gained", {})
    if any(eg.values()):
        print("\n  Endgame Intrigue / battle-icon scoring:")
        for pid, d in eg.items():
            if d:
                print(f"     P{pid}  +{d} VP")
    print_standings(gs, "FINAL STANDINGS")
    w = gs.players[gs.winner]
    print(f"\n  WINNER: P{gs.winner} ({w.leader.name}) with {w.victory_points} VP")
    top = max(p.victory_points for p in gs.players)
    if sum(1 for p in gs.players if p.victory_points == top) > 1:
        print("  (tie on VP - broken by spice, then solari, then water, then garrison)")
    print(f"  Total moves: {move}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=4, choices=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--agents", type=str, default=None)
    ap.add_argument("--real-leaders", action="store_true",
                    help="use real (unverified) Leader abilities instead of "
                         "the default no-op Leaders")
    args = ap.parse_args()
    import random
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    specs = args.agents.split(",") if args.agents else None
    run(args.players, seed, specs, neutral_leaders=not args.real_leaders)


if __name__ == "__main__":
    main()
