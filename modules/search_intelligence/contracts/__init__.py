"""Stable data contracts for the search-skill boundary."""

from .search_observation import SearchObservation, TargetDetection, Viewpoint
from .search_outcome import SearchOutcome, SearchOutcomeStatus
from .search_state import SearchState
from .search_task import (
    SearchArea,
    SearchBudget,
    SearchSuccessCriteria,
    SearchTarget,
    SearchTask,
)

__all__ = [
    "SearchArea",
    "SearchBudget",
    "SearchObservation",
    "SearchOutcome",
    "SearchOutcomeStatus",
    "SearchState",
    "SearchSuccessCriteria",
    "SearchTarget",
    "SearchTask",
    "TargetDetection",
    "Viewpoint",
]
