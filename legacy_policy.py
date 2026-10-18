"""A small mutable heuristic. Decisions use public state and the player's own cards."""
import copy
import math
import random
from game.data import BOARDS, FACTIONS, TECH, CONFLICTS, CONTRACTS, SPICE_FLOW

DEFAULT_WEIGHTS = {
    'vp': 16.0, 'spice': 1.5, 'water': 2.0, 'solari': 1.0,
    'troops': 0.8, 'draw': 1.8, 'intrigue': 1.5, 'persuasion': 0.9,
    'swords': 0.4, 'worm': 3.0, 'contract': 1.5, 'influence': 2.0,
    'spy': 1.3, 'trash': 0.8, 'swordmaster': 13.0, 'council': 7.0,
    'hooks': 4.0, 'combat': 1.0, 'deploy': 1.0, 'card': 1.0,
    'tech': 1.0, 'agent': 2.0, 'special': 0.6, 'noise': 0.10,
    'upgrade_savings': 0.5, 'faction_progress': 0.35, 'purchase_guard': 1.0,
    'deck_quality': 1.0, 'battle_icon': 0.5, 'combat_buffer': 1.5,
    'troop_cost': 1.0, 'finish': 0.4, 'flexibility': 0.7,
}


def mutate(weights, rng, sigma=0.12, max_traits=3):
    """Change only 1–3 traits; all other weights stay exactly as they were."""
    child = weights.copy()
    traits = rng.sample(list(weights), rng.randint(1, min(max_traits, len(weights))))
    for key in traits:
        child[key] = round(max(0.01, min(100, weights[key] * math.exp(rng.gauss(0, sigma)))), 6)
    return child


class Agent:
    def __init__(self, weights=None, seed=0):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.rng = random.Random(seed)

    def horizon(self, game):
        # Only deck LENGTH and public scores, never future conflict identities.
        distance = max(1, 10 - max(p['vp'] for p in game.players))
        return min(len(game.conflict_deck) + 1, max(1, distance * 1.5))

    def vp_value(self, game, pid):
        urgency = game.players[pid]['vp'] >= 8 or max(p['vp'] for p in game.players) >= 9
        return self.weights['vp'] * (1 + self.weights['finish'] * urgency)

    def upgrade_value(self, game, pid, name):
        p = game.players[pid]
        if p[name]:
            return 0
        return self.weights[name] * (0.35 + 0.2 * self.horizon(game))

    def savings(self, game, pid, solari):
        p = game.players[pid]
        value = 0
        for name, space in [('swordmaster', 'swordmaster'), ('council', 'high-council')]:
            if not p[name]:
                cost = max(1, game.board_cost(p, space).get('solari', 0))
                value += self.upgrade_value(game, pid, name) * min(1, max(0, solari)/cost)
        return self.weights['upgrade_savings'] * value

    def resource(self, game, pid, name, amount=1, economy=True):
        p = game.players[pid]
        weight = self.weights.get(name, self.weights['special'])
        if name == 'vp':
            return amount * self.vp_value(game, pid)
        if name in ('water', 'spice', 'solari'):
            weight /= 1 + p[name] / 6
        if name == 'troops' and amount > 0:
            amount = min(amount, game.supply(p))
            weight /= 1 + p['garrison']/8
        if name == 'draw' and p['has_revealed']:
            weight *= 0.5
        value = amount * weight
        if name == 'solari' and economy:
            value += self.savings(game, pid, p['solari']+amount) - self.savings(game, pid, p['solari'])
        return value

    def influence(self, game, pid, faction, amount=1):
        if faction == 'any':
            return max(self.influence(game, pid, f, amount) for f in FACTIONS)
        old = game.players[pid]['influence'][faction]
        new = max(0, min(6, old + amount))
        value = (new - old) * self.weights['influence']
        value += ((new >= 2) - (old >= 2)) * self.vp_value(game, pid)
        owner = game.alliances[faction]
        threshold = 4 if owner is None else max(4, game.players[owner]['influence'][faction] + 1)
        if owner != pid and old < threshold <= new:
            value += self.vp_value(game, pid)
        if owner == pid and new < 4:
            value -= self.vp_value(game, pid)
        # Reward partial progress toward reachable points, not just the final bump.
        targets = [2] if old < 2 or new < 2 else []
        if owner != pid and threshold <= 6:
            targets.append(threshold)
        for target in targets:
            value += self.weights['faction_progress'] * self.vp_value(game, pid) * ((min(new, target)/target)**2 - (min(old, target)/target)**2)
        return value

    def battle_value(self, game, pid, player=None):
        p = game.players[pid] if player is None else player
        if not game.conflict or not game.units(p):
            return 0
        strength = game.strength(p)
        if not p['has_revealed']:
            fraction = 1 if game.available_agents(p) == 0 else 0.5
            strength += fraction * sum(game.definition(u).get('swords', 0) for u in p['hand'])
        rivals = []
        for q in game.players:
            if q['id'] == pid:
                continue
            future = 0 if q['has_revealed'] else self.weights['combat_buffer'] * min(2, game.available_agents(q))
            rivals.append(game.strength(q) + future)
        rank = sum(r > strength for r in rivals) + int(any(r == strength and r > 0 for r in rivals))
        if strength <= 0 or rank >= 3:
            return 0
        conflict = CONFLICTS[game.conflict]
        value = self.effects(game, pid, conflict['rewards'][rank], tactical=False) * (2 if p['worms'] else 1)
        if rank == 0:
            icon = 'ornithopter' if 'ornithopter-fleet' in p['tech'] else conflict['icon']
            matches = icon != 'wild' and any(b['faceup'] and ('ornithopter' if 'ornithopter-fleet' in p['tech'] else b['icon']) == icon for b in p['battles'])
            value += self.vp_value(game, pid) * self.weights['battle_icon'] * (1 if matches else 0.2)
            if conflict['location']:
                value += self.horizon(game) * 0.3 * self.weights['spice']
        return value

    def effects(self, game, pid, effects, source=None, tactical=True):
        value = 0.0
        projected = game.players[pid].copy()
        combat_changed = False
        for effect in effects:
            op, *args = effect
            if op == 'gain':
                name, amount = args
                if tactical and name in ('swords', 'worm'):
                    if name == 'worm' and game.shield and game.conflict and CONFLICTS[game.conflict]['location']:
                        continue
                    field = 'worms' if name == 'worm' else 'swords'
                    projected[field] += amount
                    combat_changed = True
                else:
                    value += self.resource(game, pid, name, amount)
            elif op == 'influence':
                value += self.influence(game, pid, *args)
            elif op == 'when':
                if game.condition(args[0], pid, source):
                    value += self.effects(game, pid, args[1], source, tactical)
            elif op == 'pay':
                if game.players[pid][args[0]] >= args[1]:
                    value += self.effects(game, pid, args[2], source, tactical) + self.resource(game, pid, args[0], -args[1])
            elif op == 'choice':
                value += max((self.effects(game, pid, branch, source, tactical) for branch in args[0] if game.branch_available(branch, pid, source)), default=0)
            elif op in ('discard', 'trash', 'recall', 'retreat', 'selftrash'):
                if op == 'trash':
                    value += self.weights['trash']
                if args and isinstance(args[-1], (list, tuple)):
                    value += self.effects(game, pid, args[-1], source, tactical)
            elif op == 'special':
                name = args[0]
                if name == 'maker':
                    space = args[1]
                    value += self.resource(game, pid, 'spice', BOARDS.get(space, {}).get('maker', 0) + game.bonus.get(space, 0))
                elif name in ('swordmaster', 'council'):
                    value += self.upgrade_value(game, pid, name)
                elif name == 'next_combat':
                    pass
                else:
                    value += self.weights.get(name, self.weights['special'])
            elif op == 'hooks':
                value += 0 if game.players[pid]['hooks'] else self.weights['hooks'] * self.horizon(game)/5
            else:
                value += self.weights.get(op, self.weights['special'])
        if combat_changed:
            value += self.weights['combat'] * (self.battle_value(game, pid, projected) - self.battle_value(game, pid))
        return value

    def card(self, game, pid, uid):
        d = game.definition(uid)
        return (d.get('persuasion', 0) * self.weights['persuasion']
                + d.get('swords', 0) * self.weights['swords']
                + len(d.get('icons', [])) * 0.5
                + self.effects(game, pid, d.get('agent', []), uid, tactical=False) * 0.5
                + self.effects(game, pid, d.get('reveal', []), uid, tactical=False) * 0.5)

    def persuasion(self, game, pid, cards):
        p = game.players[pid]
        def gained(effects):
            total = 0
            for op, *args in effects:
                if op == 'gain' and args[0] == 'persuasion':
                    total += args[1]
                elif op == 'when' and game.condition(args[0], pid):
                    total += gained(args[1])
                elif op == 'choice':
                    total += max((gained(branch) for branch in args[0]), default=0)
            return total
        return (p['persuasion'] + 2*int(p['council']) + int(pid in game.board.get('assembly-hall', []))
                + p['navigation_bonus'] + sum(game.definition(u)['persuasion'] + gained(game.definition(u)['reveal']) for u in cards))

    def deck_average(self, game, pid):
        p = game.players[pid]
        # Aggregate owned cards; never inspect draw order or opponents' cards.
        owned = [u for zone in ('deck', 'hand', 'discard', 'played', 'revealed') for u in p[zone]]
        return sum(self.card(game, pid, u) for u in owned)/max(1, len(owned))

    def placement_context(self, game, pid, action):
        preview = copy.copy(game)
        preview.players = list(game.players)
        p = game.players[pid].copy()
        preview.players[pid] = p
        uid, space = action['card'], action['space']
        p['hand'] = [u for u in p['hand'] if u != uid]
        p['played'] = p['played'] + [uid]
        for resource, cost in game.board_cost(p, space).items() if space in BOARDS else []:
            p[resource] -= cost
        preview.ctx = {**game.ctx, 'space': space, 'placed_card': uid, 'gained_spice': 0, 'deployed': 0}
        preview.phase = 'agent'
        return preview

    def score(self, game, pid, action):
        w = self.weights
        p = game.players[pid]
        kind = action['kind']
        value = self.effects(game, pid, action.get('effects', []), action.get('source'))
        if kind == 'choose_card':
            return max(self.score(game, pid, a) for a in game.destinations(p, action['card']))
        if kind == 'place':
            space = action['space']
            board = BOARDS.get(space, {})
            uid = action['card']
            preview = self.placement_context(game, pid, action)
            value += w['agent'] + self.effects(preview, pid, board.get('effects', []))
            value += self.effects(preview, pid, game.definition(uid)['agent'], uid)
            remaining = [u for u in p['hand'] if u != uid]
            before, after = self.persuasion(game, pid, p['hand']), self.persuasion(game, pid, remaining)
            if before >= 9 > after and game.reserve.get(SPICE_FLOW):
                value -= w['purchase_guard'] * self.vp_value(game, pid)
            other_icons = {icon for u in remaining for icon in game.card_icons(p, u)}
            lost_icons = game.card_icons(p, uid) - other_icons - {board.get('icon')}
            if game.available_agents(p) > 1:
                value -= w['flexibility'] * len(lost_icons & set(FACTIONS))
            value -= game.definition(uid)['persuasion'] * w['persuasion'] * 0.25
            if board.get('faction'):
                value += self.influence(game, pid, board['faction'])
            if space in game.bonus:
                value += self.resource(game, pid, 'spice', board.get('maker', 0) + game.bonus[space])
            if space in BOARDS:
                value += sum(self.resource(game, pid, k, -n, economy=space not in ('swordmaster', 'high-council')) for k, n in game.board_cost(p, space).items())
            if board.get('combat') and p['garrison'] + p['commanders_garrison']:
                deployed = p.copy()
                count = min(2, p['garrison'])
                deployed['combat'] += count
                deployed['garrison'] -= count
                value += w['combat'] * max(0, self.battle_value(game, pid, deployed) - self.battle_value(game, pid) - count*w['troops']*w['troop_cost'])
            for contract in p['contracts']:
                if CONTRACTS[contract]['condition'] == ['space', space]:
                    value += self.effects(game, pid, CONTRACTS[contract]['effects'])
            return value
        if kind == 'resolve':
            job = game.tasks[action['index']]
            return 2 + self.effects(game, pid, [job['effect']], job['source'])
        if kind in ('nothing', 'finish_turn', 'combat_pass', 'finish_endgame'):
            return 0
        if kind == 'reveal':
            return -w['agent'] if game.available_agents(p) and any(game.destinations(p, u) for u in p['hand']) else 0
        if kind == 'influence':
            return self.influence(game, pid, action['faction'], action['amount'])
        if kind == 'transfer_influence':
            return self.influence(game, pid, action['to']) + self.influence(game, pid, action['from'], -1)
        if kind == 'deploy_turn':
            return w['deploy'] if p['garrison'] + p['commanders_garrison'] else -1
        if kind == 'deploy':
            projected = p.copy()
            projected['combat'] += action['troops']
            projected['garrison'] -= action['troops']
            projected['commanders_combat'] += action['commanders']
            projected['commanders_garrison'] -= action['commanders']
            gain = self.battle_value(game, pid, projected) - self.battle_value(game, pid)
            cost = w['troop_cost'] * w['troops'] * (action['troops'] + 0.2*action['commanders'])
            return w['combat'] * gain - cost
        if kind == 'buy_card':
            d = game.definition(action['card'])
            value += self.effects(game, pid, d.get('acquire', []), action['card'])
            quality = self.card(game, pid, action['card']) - self.deck_average(game, pid)
            value += w['card'] * w['deck_quality'] * quality * min(3, self.horizon(game)/2)
            currency = action['currency']
            if currency == 'persuasion':
                value -= action['cost']*0.05
                if game.key(action['card']) != SPICE_FLOW and p['persuasion'] >= 9 > p['persuasion']-action['cost'] and game.reserve.get(SPICE_FLOW):
                    value -= w['purchase_guard'] * self.vp_value(game, pid)
            else:
                value += self.resource(game, pid, currency, -action['cost'])
            return value
        if kind == 'buy_tech':
            d = TECH[p['secret_tech'] if action['stack'] is None else game.tech_stacks[action['stack']][-1]]
            return w['tech'] * (2 + self.effects(game, pid, d.get('acquire', []) + d.get('active', []) + d.get('endgame', []))) - action['cost'] * w['spice'] * 0.5
        if kind == 'play_intrigue':
            return self.effects(game, pid, game.definition(action['card'])['effects'], action['card']) - w['intrigue']*0.25
        if kind == 'activate_tech':
            return self.effects(game, pid, TECH[action['tech']]['active']) + 0.2
        if kind in ('discard_card', 'trash_card'):
            quality = self.card(game, pid, action['card'])
            if kind == 'trash_card':
                improvement = self.deck_average(game, pid) - quality
                value += w['trash'] * improvement * min(3, self.horizon(game)/2)
            else:
                value -= quality * 0.5
            if action['card'] in p['hand']:
                value -= game.definition(action['card']).get('persuasion', 0)*w['persuasion']*0.5
            return value
        if kind == 'pay':
            return value + self.resource(game, pid, action['resource'], -action['amount'])
        if kind == 'place_spy':
            targets = [b for b, d in BOARDS.items() if action['post'] in d['posts']]
            usefulness = max((self.upgrade_value(game, pid, 'swordmaster')/10 if b == 'swordmaster' else self.influence(game, pid, BOARDS[b]['faction'])/10 if BOARDS[b]['faction'] else 0.2 for b in targets), default=0)
            return w['spy'] * (1 + usefulness)
        if kind == 'recall':
            return value - len(action['posts']) * w['spy'] * 0.3
        if kind == 'recruit_commander':
            return w['troops'] * 3 + self.resource(game, pid, 'solari', -game.commander_cost(p))
        if kind == 'retreat':
            projected = p.copy()
            projected['combat'] -= action['troops']
            projected['commanders_combat'] -= action['commanders']
            projected['garrison'] += action['troops']
            projected['commanders_garrison'] += action['commanders']
            return value + w['combat']*(self.battle_value(game, pid, projected)-self.battle_value(game, pid)) + action['troops']*w['troops']*w['troop_cost']
        if kind == 'special_action':
            op = action['op']
            if op == 'battle_pair':
                return self.vp_value(game, pid)
            if op == 'lose_influence':
                return value + self.influence(game, pid, action['faction'], -1)
            if op == 'recall_agent':
                return w['agent']
            if op in ('take_contract', 'three_contracts'):
                contract = CONTRACTS[action['contract']]
                condition = contract['condition']
                probability = 0.4
                if condition[0] == 'space':
                    probability = 0.8 if any(a['space'] == condition[1] for u in p['hand'] for a in game.destinations(p, u)) else 0.25
                return w['contract'] + probability*self.effects(game, pid, contract['effects'])
            if op == 'discard_many':
                return value - sum(self.card(game, pid, u) for u in action['cards']) * 0.5
            return value + w.get(op, w['special'])
        return value + w['special']

    def choose(self, game):
        pid = game.pending['player']
        return max(game.legal_actions(), key=lambda a: self.score(game, pid, a) + self.rng.uniform(-self.weights['noise'], self.weights['noise']))
