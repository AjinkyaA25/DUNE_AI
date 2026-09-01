"""AI components for Dune Imperium: Uprising self-play."""
from src.ai.features import encode_state, FEATURE_DIM
from src.ai.value_model import ValueModel
from src.ai.agents import (
    Agent, RandomAgent, HeuristicAgent, GreedyValueAgent, make_agent,
)
from src.ai.opening_book import OpeningBook

__all__ = [
    "encode_state", "FEATURE_DIM",
    "ValueModel",
    "Agent", "RandomAgent", "HeuristicAgent", "GreedyValueAgent", "make_agent",
    "OpeningBook",
]
