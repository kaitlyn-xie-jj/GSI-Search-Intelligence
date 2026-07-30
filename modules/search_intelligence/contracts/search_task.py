"""Immutable task input for one active-search session."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


def _as_tuple(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,) if values.strip() else ()
    return tuple(str(value) for value in values if str(value).strip())


@dataclass(frozen=True)
class SearchTarget:
    """Open-vocabulary description of the target, without ground-truth identity."""

    query: str
    category: Optional[str] = None
    subtype: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    positive_prompts: Tuple[str, ...] = ()
    negative_prompts: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("SearchTarget.query must not be empty")
        object.__setattr__(self, "attributes", dict(self.attributes))
        object.__setattr__(self, "positive_prompts", _as_tuple(self.positive_prompts))
        object.__setattr__(self, "negative_prompts", _as_tuple(self.negative_prompts))


@dataclass(frozen=True)
class SearchArea:
    """Named search region and its platform-neutral geometry."""

    area_id: str
    geometry: Mapping[str, Any]
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.area_id.strip():
            raise ValueError("SearchArea.area_id must not be empty")
        if not isinstance(self.geometry, Mapping):
            raise TypeError("SearchArea.geometry must be a mapping")
        object.__setattr__(self, "geometry", dict(self.geometry))


@dataclass(frozen=True)
class SearchBudget:
    """Optional hard limits for a search session."""

    time_limit_s: Optional[float] = None
    distance_limit_m: Optional[float] = None
    energy_limit: Optional[float] = None
    max_viewpoints: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("time_limit_s", "distance_limit_m", "energy_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"SearchBudget.{name} must be positive")
        if self.max_viewpoints is not None and self.max_viewpoints <= 0:
            raise ValueError("SearchBudget.max_viewpoints must be positive")


@dataclass(frozen=True)
class SearchSuccessCriteria:
    """Observable conditions required to declare that a target was found."""

    min_confidence: float = 0.5
    min_confirmations: int = 1
    min_persistence_s: float = 0.0
    max_localization_error_m: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if self.min_confirmations <= 0:
            raise ValueError("min_confirmations must be positive")
        if self.min_persistence_s < 0:
            raise ValueError("min_persistence_s must not be negative")
        if self.max_localization_error_m is not None and self.max_localization_error_m < 0:
            raise ValueError("max_localization_error_m must not be negative")


@dataclass(frozen=True)
class SearchTask:
    """Complete input contract passed from GSI planning into the Search Skill."""

    task_id: str
    target: SearchTarget
    search_area: SearchArea
    instruction: str = ""
    budget: SearchBudget = field(default_factory=SearchBudget)
    success_criteria: SearchSuccessCriteria = field(default_factory=SearchSuccessCriteria)
    context_priors: Tuple[str, ...] = ()
    excluded_regions: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("SearchTask.task_id must not be empty")
        object.__setattr__(self, "context_priors", _as_tuple(self.context_priors))
        object.__setattr__(self, "excluded_regions", _as_tuple(self.excluded_regions))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_skill_params(
        cls,
        params: Mapping[str, Any],
        *,
        instruction: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SearchTask":
        """Adapt the current PlanTranslator search parameters to this contract.

        ``object_id`` and ``target_ids`` are deliberately ignored. They are simulator
        ground truth and must remain behind the observation interface.
        """
        task_id = str(params.get("task_id") or "search")
        area_token = str(params.get("area_token") or params.get("area_id") or "search-area")
        raw_area = params.get("area")
        area_geometry = dict(raw_area) if isinstance(raw_area, Mapping) else {}

        target_token = str(params.get("target_token") or "target")
        raw_target = params.get("target")
        target_data = dict(raw_target) if isinstance(raw_target, Mapping) else {}
        attributes = target_data.get("features") or target_data.get("attributes") or {}
        if not isinstance(attributes, Mapping):
            attributes = {"value": attributes}

        target = SearchTarget(
            query=target_token,
            category=target_data.get("class") or target_data.get("category"),
            subtype=target_data.get("type") or target_data.get("subtype"),
            attributes=attributes,
            positive_prompts=_as_tuple(params.get("positive_prompts")),
            negative_prompts=_as_tuple(params.get("negative_prompts")),
        )

        budget = SearchBudget(
            time_limit_s=params.get("time_budget_s") or params.get("max_search_time_s"),
            distance_limit_m=params.get("distance_budget_m"),
            energy_limit=params.get("energy_budget"),
            max_viewpoints=params.get("max_viewpoints"),
        )
        success = SearchSuccessCriteria(
            min_confidence=float(params.get("conf_ge", 0.5)),
            min_confirmations=int(params.get("min_confirmations", 1)),
            min_persistence_s=float(params.get("persist_ge_s", 0.0)),
            max_localization_error_m=params.get("max_localization_error_m"),
        )

        public_metadata = dict(metadata or {})
        for key in ("goal_type", "area_token", "target_token"):
            if key in params:
                public_metadata.setdefault(key, params[key])

        return cls(
            task_id=task_id,
            target=target,
            search_area=SearchArea(
                area_id=area_token,
                geometry=area_geometry,
                description=params.get("area_description"),
            ),
            instruction=instruction,
            budget=budget,
            success_criteria=success,
            context_priors=_as_tuple(params.get("context_priors")),
            excluded_regions=_as_tuple(params.get("excluded_regions")),
            metadata=public_metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
