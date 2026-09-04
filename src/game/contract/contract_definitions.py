"""
Complete contract cards for Dune Imperium Uprising
All contracts from the game
"""
from .contract import Contract, ContractType


def create_uprising_contracts() -> list:
    """
    Create all contract cards from Uprising

    Returns:
        List of Contract objects
    """
    contracts = []

    # ===== BOARD SPACE CONTRACTS =====
    # Complete when visiting specific board space + get bonus rewards

    # Arrakeen (Troop + Normal Spy)
    # Visit Arrakeen → Gain 1 troop, 1 spy (normal placement), complete
    arrakeen_troop = Contract(
        name="Arrakeen (Troop)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"troops": 1, "spy": 1},  # Normal spy
        trigger_condition={"board_space": "Arrakeen"}
    )
    contracts.append(arrakeen_troop)

    # Arrakeen (Water)
    # Visit Arrakeen → Gain 1 water, complete
    arrakeen_water = Contract(
        name="Arrakeen (Water)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"water": 1},
        trigger_condition={"board_space": "Arrakeen"}
    )
    contracts.append(arrakeen_water)

    # Espionage
    # Visit Espionage → Gain 3 solari, complete
    espionage = Contract(
        name="Espionage",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"solari": 3},
        trigger_condition={"board_space": "Espionage"}
    )
    contracts.append(espionage)

    # High Council (Uplift)
    # Visit High Council → Uplift one agent, complete
    high_council_uplift = Contract(
        name="High Council (Uplift)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"uplift": 1},
        trigger_condition={"board_space": "High Council"}
    )
    contracts.append(high_council_uplift)

    # High Council (Solari)
    # Visit High Council → Gain 3 solari, complete
    high_council_solari = Contract(
        name="High Council (Solari)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"solari": 3},
        trigger_condition={"board_space": "High Council"}
    )
    contracts.append(high_council_solari)

    # High Council (Bene Gesserit)
    # Visit High Council → Gain 1 Bene Gesserit influence, complete
    high_council_bene = Contract(
        name="High Council (Bene Gesserit)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"influence_bene_gesserit": 1},
        trigger_condition={"board_space": "High Council"}
    )
    contracts.append(high_council_bene)

    # Deliver Supplies (Special Spy)
    # Visit Deliver Supplies → Gain 1 solari, 1 spy (special - can place on occupied), complete
    deliver_supplies_spy = Contract(
        name="Deliver Supplies (Spy)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"solari": 1, "spy_special": 1},  # Special spy!
        trigger_condition={"board_space": "Deliver Supplies"}
    )
    contracts.append(deliver_supplies_spy)

    # Deliver Supplies (Solari)
    # Visit Deliver Supplies → Gain 3 solari, complete
    deliver_supplies_solari = Contract(
        name="Deliver Supplies (Solari)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"solari": 3},
        trigger_condition={"board_space": "Deliver Supplies"}
    )
    contracts.append(deliver_supplies_solari)

    # Heighliner (Troops)
    # Visit Heighliner → Gain 2 troops, complete
    heighliner_troops = Contract(
        name="Heighliner (Troops)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"troops": 2},
        trigger_condition={"board_space": "Heighliner"}
    )
    contracts.append(heighliner_troops)

    # Heighliner (Water) - "Wetliner"
    # Visit Heighliner → Gain 2 water, complete
    heighliner_water = Contract(
        name="Heighliner (Water)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"water": 2},
        trigger_condition={"board_space": "Heighliner"}
    )
    contracts.append(heighliner_water)

    # Research Station (Solari)
    # Visit Research Station → Gain 3 solari, complete
    research_station_solari = Contract(
        name="Research Station (Solari)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"solari": 3},
        trigger_condition={"board_space": "Research Station"}
    )
    contracts.append(research_station_solari)

    # Research Station (Solari + Normal Spy)
    # Visit Research Station → Gain 2 solari, 1 spy (normal), complete
    research_station_spy = Contract(
        name="Research Station (Spy)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"solari": 2, "spy": 1},  # Normal spy
        trigger_condition={"board_space": "Research Station"}
    )
    contracts.append(research_station_spy)

    # Sardaukar (Draw)
    # Visit Sardaukar → Draw 2 cards, complete
    sardaukar_draw = Contract(
        name="Sardaukar (Draw)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"draw": 2},
        trigger_condition={"board_space": "Sardaukar"}
    )
    contracts.append(sardaukar_draw)

    # Sardaukar (Uplift)
    # Visit Sardaukar → Uplift one agent, complete
    sardaukar_uplift = Contract(
        name="Sardaukar (Uplift)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"uplift": 1},
        trigger_condition={"board_space": "Sardaukar"}
    )
    contracts.append(sardaukar_uplift)

    # Secrets
    # Visit Secrets → Gain 2 solari, draw 1 card, complete
    secrets = Contract(
        name="Secrets",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"solari": 2, "draw": 1},
        trigger_condition={"board_space": "Secrets"}
    )
    contracts.append(secrets)

    # Spice Refinery (Troops)
    # Visit Spice Refinery → Gain 2 troops, complete
    spice_refinery_troops = Contract(
        name="Spice Refinery (Troops)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"troops": 2},
        trigger_condition={"board_space": "Spice Refinery"}
    )
    contracts.append(spice_refinery_troops)

    # Spice Refinery (Draw)
    # Visit Spice Refinery → Draw 2 cards, complete
    spice_refinery_draw = Contract(
        name="Spice Refinery (Draw)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"draw": 2},
        trigger_condition={"board_space": "Spice Refinery"}
    )
    contracts.append(spice_refinery_draw)

    # Spice Refinery (Water)
    # Visit Spice Refinery → Gain 1 water, complete
    spice_refinery_water = Contract(
        name="Spice Refinery (Water)",
        contract_type=ContractType.BOARD_SPACE,
        rewards={"water": 1},
        trigger_condition={"board_space": "Spice Refinery"}
    )
    contracts.append(spice_refinery_water)

    # ===== HARVEST CONTRACTS =====
    # Complete when gaining 3+ or 4+ spice at maker spaces
    # (Hagga Basin, Imperial Basin, Deep Desert)

    # Harvest 4+ (Solari)
    # At maker space, gain 4+ spice → Gain 4 solari, complete
    harvest_4_solari = Contract(
        name="Harvest 4+ (Solari)",
        contract_type=ContractType.HARVEST,
        rewards={"solari": 4},
        trigger_condition={"min_spice": 4}
    )
    contracts.append(harvest_4_solari)

    # Harvest 4+ (Solari + Spy)
    # At maker space, gain 4+ spice → Gain 3 solari, 1 spy (normal), complete
    harvest_4_spy = Contract(
        name="Harvest 4+ (Spy)",
        contract_type=ContractType.HARVEST,
        rewards={"solari": 3, "spy": 1},
        trigger_condition={"min_spice": 4}
    )
    contracts.append(harvest_4_spy)

    # Harvest 3+ (Solari)
    # At maker space, gain 3+ spice → Gain 3 solari, complete
    harvest_3_solari = Contract(
        name="Harvest 3+ (Solari)",
        contract_type=ContractType.HARVEST,
        rewards={"solari": 3},
        trigger_condition={"min_spice": 3}
    )
    contracts.append(harvest_3_solari)

    # Harvest 3+ (Solari + Spy)
    # At maker space, gain 3+ spice → Gain 2 solari, 1 spy (normal), complete
    harvest_3_spy = Contract(
        name="Harvest 3+ (Spy)",
        contract_type=ContractType.HARVEST,
        rewards={"solari": 2, "spy": 1},
        trigger_condition={"min_spice": 3}
    )
    contracts.append(harvest_3_spy)

    # ===== ACQUIRE CARD CONTRACTS =====
    # Complete when acquiring specific card

    # The Spice Must Flow
    # Acquire "The Spice Must Flow" → Gain 3 solari, 1 Spacing Guild influence, complete
    smf = Contract(
        name="Acquire The Spice Must Flow",
        contract_type=ContractType.ACQUIRE_CARD,
        rewards={"solari": 3, "influence_spacing_guild": 1},
        trigger_condition={"card_name": "The Spice Must Flow"}
    )
    contracts.append(smf)

    # ===== ALLIANCE CONTRACTS =====
    # Complete when gaining any alliance (reaching influence 4)

    # Earn Any Alliance
    # Gain any alliance → Gain 2 troops, complete
    alliance = Contract(
        name="Earn Any Alliance",
        contract_type=ContractType.ALLIANCE,
        rewards={"troops": 2},
        trigger_condition={}
    )
    contracts.append(alliance)

    # ===== IMMEDIATE CONTRACTS =====
    # Complete immediately when taken

    # Immediate (Solari)
    # Take this contract → Gain 2 solari immediately, complete
    immediate_solari = Contract(
        name="Immediate (Solari)",
        contract_type=ContractType.IMMEDIATE,
        rewards={"solari": 2},
        trigger_condition={}
    )
    contracts.append(immediate_solari)

    # Immediate (Intrigue Trade)
    # Take this contract → Trash 1 intrigue to gain 1 intrigue and draw 1 card, complete
    # Requires player to have at least 1 intrigue
    immediate_intrigue = Contract(
        name="Immediate (Intrigue Trade)",
        contract_type=ContractType.IMMEDIATE,
        rewards={"trash_intrigue_for": {"intrigue": 1, "draw": 1}},
        trigger_condition={"requires_intrigue": True}
    )
    contracts.append(immediate_intrigue)

    return contracts


def get_contract_by_name(name: str) -> Contract:
    """
    Get a specific contract by name

    Args:
        name: Contract name

    Returns:
        Contract object or None if not found
    """
    contracts = create_uprising_contracts()
    for contract in contracts:
        if contract.name == name:
            return contract
    return None

# Total contracts: 26
# - Board Space: 17 contracts (removed the fabricated "Spice Refinery
#   (Draw + Troops)" — no such contract exists)
# - Harvest: 4 contracts (all regular Spies, not special)
# - Acquire Card: 1 contract
# - Alliance: 1 contract
# - Immediate: 2 contracts
# - Special Spy contracts: 1 (Deliver Supplies Spy)
# - Normal Spy contracts: 4 (Arrakeen Troop, Research Station Spy,
#   Harvest 4+ Spy, Harvest 3+ Spy)