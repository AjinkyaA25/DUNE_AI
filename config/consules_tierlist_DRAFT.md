# Consules "Ranking EVERY Uprising Card" — tier list

Source: youtu.be/aDNenYScyhI (Consules / Dinosaur11), 2h11m, community-assembled.
Board on screen: **Guild Spy (alone, #1) → S → A → B → C → D**  (no F tier).

These values live in `src/data/card_definitions.py` -> `_IMPERIUM_TIERS` and are
applied to each `Card.tier`. `agents._acquire_card_value` blends the anchor
(`TIER_ANCHOR`) with the situational score at 0.7 / 0.3. Edit the dict there to
change a rating; the trained value net learns deviations on top.

## Applied (42 cards)

**S:** Guild Spy, Undercover Asset, Calculus of Power, Desert Power, Overthrow,
Public Spectacle, Strike Fleet, Interstellar Trade

**A:** Space-Time Folding, Spy Network, Imperial Spymaster, Reliable Informant,
Wheels Within Wheels, Double Agent, Chani, Dangerous Rhetoric, Captured Mentat,
Corrinth City, Steersman, Sardaukar Coordination, Spacing Guild's Favor,
Truthtrance, Price is No Object, Ecological Testing Station, Treacherous Maneuver

**B:** Sardaukar Soldier, Weirding Woman, Maker Keeper, Guild Envoy,
Covert Operation, In High Places, Long Live the Fighters, Leadership,
Bene Gesserit Operative, Prepare the Way (reserve)

**C:** Unswerving Loyalty, Fedaykin Stilltent, Cargo Runner, Delivery Agreement,
Branching Path

**D:** Smuggler's Harvester, Hidden Missive, Desert Survival

### Confidence
Rows 1-cost → 3-cost (through Calculus of Power) are verbatim from the transcript.
The rest (Overthrow, Public Spectacle, Strike Fleet, Interstellar Trade S;
Steersman, Sardaukar Coordination, Spacing Guild's Favor, Truthtrance,
Price is No Object, Ecological Testing Station, Treacherous Maneuver A;
In High Places, Long Live the Fighters, Leadership, Bene Gesserit Operative B)
are keyword-parse + YouTube-comment corroboration — adjust freely.

## Deliberately NOT rated — AI self-assesses (15 cards)

Arrakis Revolt, Junction Headquarters, Maula Pistol, Northern Watermaster,
Paracompass, Priority Contracts, Rebel Supplier, Shishakli, Smuggler's Haven,
Southern Elders, Stilgar the Devoted, Subversive Advisor, The Beast's Spoils,
Thumper, Tread in Darkness.

`tier=None` → scored purely on situational effect value, so the AI decides
where they land relative to what the rest of the list values.
