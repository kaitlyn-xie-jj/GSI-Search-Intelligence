"""Task-conditioned semantic prior contract and grid projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .contracts import SearchTask
from .search_space import SearchGrid
from .semantic_map import normalize_semantic_label


@dataclass(frozen=True)
class SearchPriorRequest:
    """Compact task and semantic-map context supplied to the upper-level LLM."""

    task_id: str
    target_query: str
    semantic_inventory: Mapping[str, int]
    instruction: str = ""
    target_category: Optional[str] = None
    target_subtype: Optional[str] = None
    target_attributes: Mapping[str, Any] = field(default_factory=dict)
    context_priors: Tuple[str, ...] = ()
    excluded_regions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.target_query.strip():
            raise ValueError("prior request task ID and target query must not be empty")
        inventory = {
            normalize_semantic_label(label): int(count)
            for label, count in self.semantic_inventory.items()
            if normalize_semantic_label(label) and int(count) > 0
        }
        object.__setattr__(self, "semantic_inventory", inventory)
        object.__setattr__(self, "target_attributes", dict(self.target_attributes))
        object.__setattr__(self, "context_priors", tuple(self.context_priors))
        object.__setattr__(self, "excluded_regions", tuple(self.excluded_regions))

    @classmethod
    def from_task_and_grid(
        cls,
        task: SearchTask,
        grid: SearchGrid,
    ) -> "SearchPriorRequest":
        inventory: Dict[str, int] = {}
        for cell in grid.searchable_cells:
            for label in cell.semantic_labels:
                normalized = normalize_semantic_label(label)
                if normalized:
                    inventory[normalized] = inventory.get(normalized, 0) + 1
        return cls(
            task_id=task.task_id,
            target_query=task.target.query,
            semantic_inventory=inventory,
            instruction=task.instruction,
            target_category=task.target.category,
            target_subtype=task.target.subtype,
            target_attributes=task.target.attributes,
            context_priors=task.context_priors,
            excluded_regions=task.excluded_regions,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchPriorProjection:
    """Auditable cell-level result of projecting one semantic SearchPrior."""

    task_id: str
    belief: Mapping[str, float]
    raw_cell_scores: Mapping[str, float]
    matched_labels: Tuple[str, ...] = ()
    unmatched_labels: Tuple[str, ...] = ()
    confidence: float = 1.0
    projection_mode: str = "cell_affinity"

    def __post_init__(self) -> None:
        object.__setattr__(self, "belief", dict(self.belief))
        object.__setattr__(self, "raw_cell_scores", dict(self.raw_cell_scores))
        object.__setattr__(self, "matched_labels", tuple(self.matched_labels))
        object.__setattr__(self, "unmatched_labels", tuple(self.unmatched_labels))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchPrior:
    """Validated LLM output describing likely semantic target regions."""

    task_id: str
    semantic_weights: Mapping[str, float]
    confidence: float = 1.0
    default_weight: float = 0.0
    excluded_labels: Tuple[str, ...] = ()
    projection_mode: str = "cell_affinity"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("SearchPrior.task_id must not be empty")
        weights = {
            normalize_semantic_label(label): float(weight)
            for label, weight in self.semantic_weights.items()
            if normalize_semantic_label(label)
        }
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("semantic weights must not be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SearchPrior.confidence must be within [0, 1]")
        if self.default_weight < 0:
            raise ValueError("SearchPrior.default_weight must not be negative")
        if self.projection_mode not in {"cell_affinity", "label_mass"}:
            raise ValueError(
                "SearchPrior.projection_mode must be cell_affinity or label_mass"
            )
        object.__setattr__(self, "semantic_weights", weights)
        object.__setattr__(
            self,
            "excluded_labels",
            tuple(dict.fromkeys(
                normalize_semantic_label(label)
                for label in self.excluded_labels
                if normalize_semantic_label(label)
            )),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_llm_output(
        cls,
        task_id: str,
        output: Mapping[str, Any],
    ) -> "SearchPrior":
        """Adapt the agreed JSON-shaped LLM output into a validated contract."""
        raw_weights = (
            output.get("semantic_weights")
            or output.get("semantic_region_weights")
            or output.get("region_weights")
            or {}
        )
        if not isinstance(raw_weights, Mapping):
            raise TypeError("LLM semantic weights must be a mapping")
        raw_exclusions = output.get("excluded_labels") or output.get("negative_regions") or ()
        if isinstance(raw_exclusions, str):
            raw_exclusions = (raw_exclusions,)
        return cls(
            task_id=task_id,
            semantic_weights=raw_weights,
            confidence=float(output.get("confidence", 1.0)),
            default_weight=float(output.get("default_weight", 0.0)),
            excluded_labels=tuple(raw_exclusions),
            projection_mode=str(output.get("projection_mode", "cell_affinity")),
            metadata=output.get("metadata") or {},
        )

    @staticmethod
    def llm_output_schema() -> Dict[str, Any]:
        """Return the JSON schema expected from the upper-level LLM."""
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["semantic_weights", "confidence"],
            "properties": {
                "semantic_weights": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0.0},
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "default_weight": {"type": "number", "minimum": 0.0},
                "excluded_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "projection_mode": {
                    "type": "string",
                    "enum": ["cell_affinity", "label_mass"],
                    "default": "cell_affinity",
                },
                "metadata": {"type": "object"},
            },
        }

    def project(self, grid: SearchGrid) -> SearchPriorProjection:
        """Project semantic weights to cells and mix with uniform uncertainty."""
        searchable = grid.searchable_cells
        if not searchable:
            return SearchPriorProjection(
                task_id=self.task_id,
                belief={},
                raw_cell_scores={},
                unmatched_labels=tuple(self.semantic_weights),
                confidence=self.confidence,
                projection_mode=self.projection_mode,
            )

        available_labels = {
            normalize_semantic_label(label)
            for cell in searchable
            for label in cell.semantic_labels
        }
        requested_labels = set(self.semantic_weights)
        matched = tuple(sorted(requested_labels & available_labels))
        unmatched = tuple(sorted(requested_labels - available_labels))
        excluded = set(self.excluded_labels)

        cell_labels: Dict[str, set[str]] = {}
        eligible_cell_ids = []
        for cell in searchable:
            labels = {normalize_semantic_label(label) for label in cell.semantic_labels}
            cell_labels[cell.cell_id] = labels
            if not labels & excluded:
                eligible_cell_ids.append(cell.cell_id)

        # An invalid all-excluded request falls back to the whole searchable grid.
        if not eligible_cell_ids:
            eligible_cell_ids = [cell.cell_id for cell in searchable]
        eligible = set(eligible_cell_ids)
        label_counts = {
            label: sum(
                cell_id in eligible and label in labels
                for cell_id, labels in cell_labels.items()
            )
            for label in self.semantic_weights
        }
        raw_scores: Dict[str, float] = {}
        for cell in searchable:
            if cell.cell_id not in eligible:
                raw_scores[cell.cell_id] = 0.0
                continue
            matching_weights = []
            for label, weight in self.semantic_weights.items():
                if label not in cell_labels[cell.cell_id]:
                    continue
                if self.projection_mode == "label_mass":
                    matching_weights.append(weight / max(1, label_counts[label]))
                else:
                    matching_weights.append(weight)
            default_score = float(self.default_weight)
            if self.projection_mode == "label_mass":
                default_score /= len(eligible)
            raw_scores[cell.cell_id] = max([default_score, *matching_weights])

        score_total = sum(
            score for cell_id, score in raw_scores.items() if cell_id in eligible
        )
        uniform_probability = 1.0 / len(eligible)
        if score_total <= 0:
            semantic_distribution = {
                cell.cell_id: uniform_probability if cell.cell_id in eligible else 0.0
                for cell in searchable
            }
        else:
            semantic_distribution = {
                cell_id: score / score_total if cell_id in eligible else 0.0
                for cell_id, score in raw_scores.items()
            }
        belief = {
            cell.cell_id: (
                self.confidence * semantic_distribution[cell.cell_id]
                + (1.0 - self.confidence)
                * (uniform_probability if cell.cell_id in eligible else 0.0)
            )
            for cell in searchable
        }
        return SearchPriorProjection(
            task_id=self.task_id,
            belief=belief,
            raw_cell_scores=raw_scores,
            matched_labels=matched,
            unmatched_labels=unmatched,
            confidence=self.confidence,
            projection_mode=self.projection_mode,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
