"""Load public SearchWorld semantics and task-conditioned prior for ROS search."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from modules.search_intelligence import (
    SearchGrid,
    SearchPrior,
    SearchTask,
    SemanticGridBuilder,
)


@dataclass(frozen=True)
class SearchScenarioContext:
    grid: SearchGrid
    initial_belief: Mapping[str, float]
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)


def load_search_scenario_context(
    task: SearchTask,
    grid: SearchGrid,
    *,
    semantic_map_path: str = "",
    search_prior_path: str = "",
) -> SearchScenarioContext:
    """Apply optional public semantics and prior without reading ground truth."""
    metadata: Dict[str, Any] = {"semantic_map_loaded": False, "prior_loaded": False}
    annotated = grid

    if semantic_map_path.strip():
        path = _resolve_existing_path(semantic_map_path)
        semantic_document = _read_mapping(path)
        nodes = _scene_nodes(semantic_document)
        annotated = SemanticGridBuilder().annotate(grid, nodes)
        metadata.update({
            "semantic_map_loaded": True,
            "semantic_map_path": str(path),
            "semantic_feature_count": len(nodes),
            "searchable_cell_count": len(annotated.searchable_cells),
        })

    initial_belief = annotated.uniform_belief()
    if search_prior_path.strip():
        path = _resolve_existing_path(search_prior_path)
        prior_document = _read_mapping(path)
        projection = SearchPrior.from_llm_output(task.task_id, prior_document).project(
            annotated
        )
        initial_belief = projection.belief
        metadata.update({
            "prior_loaded": True,
            "search_prior_path": str(path),
            "prior_confidence": projection.confidence,
            "prior_matched_labels": projection.matched_labels,
            "prior_unmatched_labels": projection.unmatched_labels,
        })

    return SearchScenarioContext(
        grid=annotated,
        initial_belief=initial_belief,
        policy_metadata=metadata,
    )


def _resolve_existing_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Search scenario artifact not found: {path}")
    return path


def _read_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise TypeError(f"Search scenario artifact must be a JSON object: {path}")
    return value


def _scene_nodes(document: Mapping[str, Any]):
    nodes = document.get("nodes")
    if nodes is None and isinstance(document.get("scene_graph"), Mapping):
        nodes = document["scene_graph"].get("nodes")
    if not isinstance(nodes, list):
        raise TypeError("semantic map must provide a nodes array")
    return nodes
