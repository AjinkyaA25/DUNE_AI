#!/usr/bin/env python
"""
Dune Imperium: Uprising — Interactive CLI Game Runner

Usage:
    python play_game.py                    # 2 players, you are Player 0
    python play_game.py --players 4        # 4-player game
    python play_game.py --seed 42          # fixed seed (reproducible)
    python play_game.py --human 1          # you are Player 1
    python play_game.py --log games/       # save game log to games/ folder

Every move is recorded. At game end the full game log is saved as JSON,
ready for ML training data.
"""
from __future__ import annotations

import argparse
import json
import os
import random as pyrandom
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from src.data.card_definitions import setup_game
from src.game.gameState import GameAction, ActionType, GameState, Phase


# ---------------------------------------------------------------------------
# Action → human-readable string
# ---------------------------------------------------------------------------

def action_label(action: GameAction) -> str:
    at = action.action_type
    if at == ActionType.AGENT_TURN:
        mods = []
        if action.use_gather_intelligence:
            mods.append("Gather Intelligence")
        if action.use_infiltrate:
            mods.append("Infiltrate")
        if action.space_option:
            mods.append(action.space_option)
        mod_str = f"  [{', '.join(mods)}]" if mods else ""
        return f"Play '{action.card_name}' → {action.space_name}{mod_str}"
    if at == ActionType.REVEAL_TURN:
        return "Reveal hand (enter buying phase)"
    if at == ActionType.PLAY_INTRIGUE:
        return f"Play Intrigue '{action.intrigue_card_name}'"
    if at == ActionType.RESOLVE_DEPLOY:
        return f"Deploy {action.deploy_count} troop(s) to the Conflict"
    if at == ActionType.RESOLVE_TRASH:
        return (f"Trash '{action.trash_card_name}'" if action.trash_card_name
                else "Trash nothing")
    if at == ActionType.RESOLVE_INFLUENCE:
        return f"Gain 1 influence with {action.influence_faction}"
    if at == ActionType.ACQUIRE_CARD:
        return f"Buy '{action.acquire_card_name}' from Imperium Row"
    if at == ActionType.ACQUIRE_RESERVE:
        label = "Prepare the Way" if action.reserve_type == "prepare_the_way" else "Spice Must Flow"
        return f"Buy reserve: {label}"
    if at == ActionType.END_REVEAL:
        return "Done buying — end turn"
    if at == ActionType.COMBAT_PASS:
        return "Pass combat (no intrigue to play)"
    if at == ActionType.RESOLVE_SPY:
        return f"Place spy at '{action.spy_post_name}'"
    if at == ActionType.RESOLVE_UPLIFT:
        return f"Uplift agent from '{action.uplift_space_name}'"
    if at == ActionType.RESOLVE_INTRIGUE_TRASH:
        return f"Trash intrigue '{action.intrigue_card_name}' for benefit"
    if at == ActionType.NO_OP:
        return "Wait"
    return repr(action)


def action_to_dict(action: GameAction) -> Dict:
    return {
        "action_type": action.action_type.value,
        "player_id":   action.player_id,
        "card_name":   action.card_name,
        "space_name":  action.space_name,
        "space_option":            action.space_option,
        "deploy_count":            action.deploy_count,
        "trash_card_name":         action.trash_card_name,
        "influence_faction":       action.influence_faction,
        "use_gather_intelligence": action.use_gather_intelligence,
        "use_infiltrate":          action.use_infiltrate,
        "spy_post_name":           action.spy_post_name,
        "uplift_space_name":       action.uplift_space_name,
        "intrigue_card_name":      action.intrigue_card_name,
        "acquire_card_name":       action.acquire_card_name,
        "reserve_type":            action.reserve_type,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEP  = "-" * 70
SEP2 = "=" * 70


def _inf_str(player) -> str:
    factions = [("EMP", "emperor"), ("SPA", "spacing_guild"),
                ("BEN", "bene_gesserit"), ("FRE", "fremen")]
    return "  ".join(f"{tag}:{player.influence[f]}" for tag, f in factions)


def display_state(gs: GameState, human_id: int) -> None:
    print(f"\n{SEP2}")
    print(f"  DUNE IMPERIUM: UPRISING  |  Round {gs.round}/10  |  {gs.phase.value.upper()}")
    print(SEP2)

    # ── Conflict ──────────────────────────────────────────────────────────
    if gs.current_conflict:
        cc = gs.current_conflict
        loc_str = ""
        if cc.location:
            ctrl = gs.controlled_by.get(cc.location)
            loc_str = f"  Location: {cc.location} → " + (f"P{ctrl}" if ctrl is not None else "uncontrolled")
        print(f"\n  Conflict: {cc.name}  (Level {cc.conflict_level}){loc_str}")
        print(f"    1st: {cc.first_place_reward}")
        print(f"    2nd: {cc.second_place_reward}")
        if cc.third_place_reward:
            print(f"    3rd: {cc.third_place_reward}")

    # ── Board ─────────────────────────────────────────────────────────────
    occupied = {s: p for s, p in gs.agent_on_space.items() if p is not None}
    if occupied:
        cols = "  |  ".join(f"{s}: P{p}" for s, p in sorted(occupied.items()))
        print(f"\n  Occupied: {cols}")

    maker_bonus = {s: b for s, b in gs.maker_bonus_spice.items() if b > 0}
    if maker_bonus:
        print(f"  Maker bonus spice: {maker_bonus}")

    # ── Imperium row ──────────────────────────────────────────────────────
    print(f"\n  Imperium Row:")
    for card in gs.imperium_row:
        syms = "/".join(s.value[:3].upper() for s in sorted(card.access_symbols, key=lambda x: x.value))
        eff_parts = []
        if card.agent_effects:
            eff_parts.append(f"agent:{card.agent_effects}")
        if card.persuasion:
            eff_parts.append(f"P:{card.persuasion}")
        if card.swords:
            eff_parts.append(f"Sw:{card.swords}")
        if card.reveal_effects:
            eff_parts.append(f"rev:{card.reveal_effects}")
        print(f"    [{card.cost:2d}]  {card.name:<22} [{syms}]  {', '.join(eff_parts)}")
    if gs.reserve_prepare_the_way:
        r = gs.reserve_prepare_the_way[-1]
        print(f"    [{r.cost:2d}]  Prepare the Way        [RESERVE] ×{len(gs.reserve_prepare_the_way)}")
    if gs.reserve_spice_must_flow:
        r = gs.reserve_spice_must_flow[-1]
        print(f"    [{r.cost:2d}]  Spice Must Flow        [RESERVE] ×{len(gs.reserve_spice_must_flow)}")

    # ── Players ───────────────────────────────────────────────────────────
    print(f"\n  Players:")
    for p in gs.players:
        you   = " <-- YOU" if p.id == human_id else ""
        rev   = " [REVEALED]" if p.id in gs.players_revealed else ""
        buy   = " [BUYING]"   if gs.player_in_reveal_buy == p.id else ""
        print(f"\n  P{p.id}{you}{rev}{buy}")
        print(f"    VP:{p.victory_points}  Sol:{p.solari}  Spi:{p.spice}  Wat:{p.water}")
        print(f"    Agents:{p.agents_available}/{p.agents_total}  "
              f"Garrison:{p.troops_garrison}  Supply:{p.troops_supply}  "
              f"Spies:{p.spies_available}/3")
        print(f"    Influence: {_inf_str(p)}")
        t_in_conflict  = gs.troops_in_conflict.get(p.id, 0)
        sw_in_conflict = gs.sandworms_in_conflict.get(p.id, 0)
        if t_in_conflict or sw_in_conflict:
            print(f"    In conflict: {t_in_conflict} troops  {sw_in_conflict} sandworms"
                  f"  (strength {gs.combat_strength.get(p.id, 0)})")
        hand_size = len(p.hand)
        discard_names = [c.name for c in p.discard]
        if p.id == human_id:
            hand_names = [c.name for c in p.hand]
            print(f"    Hand ({hand_size}): {', '.join(hand_names) if hand_names else '(empty)'}")
            in_play = [c.name for c in p.in_play]
            if in_play:
                print(f"    In play: {', '.join(in_play)}")
            if p.intrigue_cards:
                print(f"    Intrigue ({len(p.intrigue_cards)}): {', '.join(ic.name for ic in p.intrigue_cards)}")
            if gs.player_in_reveal_buy == p.id:
                print(f"    >>> Persuasion to spend: {gs.persuasion_pool[p.id]} <<<")
        else:
            print(f"    Hand size: {hand_size}  Discard: {len(discard_names)}")

    print()


def display_actions(actions: List[GameAction]) -> None:
    print(f"  {SEP}")
    print("  Valid actions:")
    for i, a in enumerate(actions):
        print(f"    [{i}] {action_label(a)}")
    print(f"  {SEP}")


# ---------------------------------------------------------------------------
# AI: random non-NO_OP action (or NO_OP if that's the only option)
# ---------------------------------------------------------------------------

def ai_action(gs: GameState, player_id: int, agent=None) -> GameAction:
    actions = gs.get_valid_actions(player_id)
    if agent is not None:
        return agent.select_action(gs, player_id, actions)
    non_noop = [a for a in actions if a.action_type != ActionType.NO_OP]
    return pyrandom.choice(non_noop or actions)


# ---------------------------------------------------------------------------
# Human input
# ---------------------------------------------------------------------------

def human_action(gs: GameState, player_id: int) -> GameAction:
    actions = gs.get_valid_actions(player_id)
    display_actions(actions)

    while True:
        try:
            raw = input("  Choose action number: ").strip()
            if raw.lower() in ("q", "quit"):
                print("Quitting game.")
                sys.exit(0)
            idx = int(raw)
            if 0 <= idx < len(actions):
                return actions[idx]
            print(f"  Please enter a number between 0 and {len(actions)-1}.")
        except (ValueError, EOFError):
            print("  Invalid input. Enter a number (or 'q' to quit).")


# ---------------------------------------------------------------------------
# Game log helpers
# ---------------------------------------------------------------------------

def save_log(log: Dict, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    gid = log["game_id"][:8]
    filename = os.path.join(log_dir, f"game_{ts}_{gid}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return filename


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def _build_agents(num_players: int, human_id: int, specs, seed):
    """specs: list of agent-spec strings per seat, or None -> 'heuristic'."""
    from src.ai.agents import make_agent
    agents = {}
    for i in range(num_players):
        if i == human_id:
            agents[i] = None
            continue
        spec = "heuristic"
        if specs and i < len(specs) and specs[i]:
            spec = specs[i]
        agents[i] = make_agent(spec, seed=(seed or 0) * 8 + i)
    return agents


def run_game(
    num_players: int  = 2,
    human_id: int     = 0,
    seed: Optional[int] = None,
    log_dir: str      = "game_logs",
    show_ai_moves: bool = True,
    agent_specs=None,
) -> Dict:
    """
    Play one full game.  Returns the game log dict.

    human_id: which player index is the human (-1 = fully AI / self-play)
    agent_specs: optional list of per-seat agent specs (see src.ai.agents.make_agent)
    """
    actual_seed = seed if seed is not None else pyrandom.randint(0, 999_999)
    pyrandom.seed(actual_seed)

    gs = setup_game(num_players=num_players, seed=actual_seed)
    agents = _build_agents(num_players, human_id, agent_specs, actual_seed)

    game_id = str(uuid.uuid4())
    log: Dict = {
        "game_id":     game_id,
        "timestamp":   datetime.now().isoformat(),
        "seed":        actual_seed,
        "num_players": num_players,
        "human_id":    human_id,
        "winner":      None,
        "final_vp":    [],
        "moves":       [],
    }

    move_num = 0

    print(f"\n{'='*70}")
    print(f"  Game ID: {game_id[:8]}   Seed: {actual_seed}   Players: {num_players}")
    print(f"  You are Player {human_id}" if human_id >= 0 else "  Self-play (all AI)")
    print(f"{'='*70}")

    while not gs.game_over:
        # ── Determine whose turn it is ────────────────────────────────────
        if gs.player_in_reveal_buy is not None:
            active = gs.player_in_reveal_buy
        else:
            active = gs.get_current_player_id()

        is_human = (active == human_id)

        # Show state only for human turns (or always if show_ai_moves)
        if is_human or show_ai_moves:
            display_state(gs, human_id)

        # ── Pick action ───────────────────────────────────────────────────
        state_before = gs.get_state_dict()

        if is_human:
            action = human_action(gs, active)
        else:
            action = ai_action(gs, active, agents.get(active))
            if show_ai_moves:
                print(f"  AI (P{active}): {action_label(action)}")

        # ── Step ──────────────────────────────────────────────────────────
        state_after, reward, done, info = gs.step(action)

        if info.get("error"):
            print(f"  [!] Error: {info['error']}")

        # ── Log move ──────────────────────────────────────────────────────
        log["moves"].append({
            "move_num":    move_num,
            "round":       state_before["round"],
            "phase":       state_before["phase"],
            "active_player": active,
            "is_human":    is_human,
            "state_before": state_before,
            "action":      action_to_dict(action),
            "state_after": state_after,
            "reward":      reward,
            "error":       info.get("error"),
        })
        move_num += 1

        # Safety: break if game somehow loops
        if move_num > 5000:
            print("  [!] Move limit reached — aborting.")
            break

    # ── Game over ─────────────────────────────────────────────────────────
    display_state(gs, human_id)

    winner = gs.winner
    final_vp = [p.victory_points for p in gs.players]
    log["winner"]   = winner
    log["final_vp"] = final_vp

    print(f"\n{'='*70}")
    print("  GAME OVER")
    for i, vp in enumerate(final_vp):
        tag = " <-- WINNER" if i == winner else ""
        you = " (you)" if i == human_id else ""
        print(f"  Player {i}{you}: {vp} VP{tag}")
    print(f"{'='*70}\n")

    saved = save_log(log, log_dir)
    print(f"  Game log saved to: {saved}")
    print(f"  Total moves recorded: {len(log['moves'])}\n")

    return log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dune Imperium: Uprising — Interactive CLI")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4],
                        help="Number of players (default: 2)")
    parser.add_argument("--human", type=int, default=0,
                        help="Which player index is human (-1 for fully AI, default: 0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--log", type=str, default="game_logs",
                        help="Directory to save game logs (default: game_logs/)")
    parser.add_argument("--quiet-ai", action="store_true",
                        help="Hide AI move announcements")
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated per-seat agent specs, e.g. "
                             "'heuristic,value:models/value_best.npz,random,heuristic'. "
                             "Seats not listed default to 'heuristic'.")
    args = parser.parse_args()

    specs = args.agents.split(",") if args.agents else None

    run_game(
        num_players=args.players,
        human_id=args.human,
        seed=args.seed,
        log_dir=args.log,
        show_ai_moves=not args.quiet_ai,
        agent_specs=specs,
    )


if __name__ == "__main__":
    main()
