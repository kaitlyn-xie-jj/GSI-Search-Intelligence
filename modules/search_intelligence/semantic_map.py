"""Projection of GSI scene-graph semantics onto the discrete search grid."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .search_space import SearchGrid, point_in_search_geometry


DEFAULT_ENVIRONMENT_CATEGORIES = frozenset({
    "area",
    "building",
    "district",
    "poi",
    "trans_facility",
    "transportation_facility",
})


def normalize_semantic_label(value: Any) -> str:
    """Normalize scene and LLM labels to a shared matching key."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_")


@dataclass(frozen=True)
class SemanticFeature:
    """Environment feature extracted from a public scene-graph node."""

    feature_id: str
    category: str
    feature_type: str
    label: str
    geometry: Mapping[str, Any]
    passability: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_id.strip():
            raise ValueError("SemanticFeature.feature_id must not be empty")
        object.__setattr__(self, "category", normalize_semantic_label(self.category))
        object.__setattr__(self, "feature_type", normalize_semantic_label(self.feature_type))
        object.__setattr__(self, "label", normalize_semantic_label(self.label))
        object.__setattr__(self, "geometry", dict(self.geometry))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def semantic_labels(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(
            label
            for label in (self.category, self.feature_type, self.label)
            if label
        ))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticGridBuilder:
    """Annotate SearchGrid cells using environment nodes from GSI's scene graph."""

    allowed_categories: frozenset[str] = DEFAULT_ENVIRONMENT_CATEGORIES
    line_buffer_m: float = 10.0
    point_buffer_m: float = 10.0
    blocked_passability: frozenset[str] = frozenset({
        "blocked",
        "impassable",
        "restricted",
    })

    def __post_init__(self) -> None:
        if self.line_buffer_m <= 0 or self.point_buffer_m <= 0:
            raise ValueError("semantic feature buffers must be positive")
        object.__setattr__(
            self,
            "allowed_categories",
            frozenset(normalize_semantic_label(item) for item in self.allowed_categories),
        )
        object.__setattr__(
            self,
            "blocked_passability",
            frozenset(normalize_semantic_label(item) for item in self.blocked_passability),
        )

    def from_scene_graph(self, grid: SearchGrid, scene_graph: Any) -> SearchGrid:
        """Read nodes through AbstractSceneGraph's public query interface."""
        nodes = getattr(scene_graph, "get_all_nodes", lambda: ())() or ()
        return self.annotate(grid, nodes)

    def annotate(
        self,
        grid: SearchGrid,
        scene_nodes: Iterable[Mapping[str, Any]],
    ) -> SearchGrid:
        features = self.extract_features(scene_nodes)
        updated_cells = []
        for cell in grid.cells:
            matching = tuple(
                feature
                for feature in features
                if point_in_search_geometry(cell.center, feature.geometry)
            )
            labels = tuple(dict.fromkeys(
                label
                for feature in matching
                for label in feature.semantic_labels
            ))
            blocked_by = tuple(
                feature.feature_id
                for feature in matching
                if normalize_semantic_label(feature.passability or "")
                in self.blocked_passability
            )
            metadata = dict(cell.metadata)
            metadata.update({
                "semantic_feature_ids": tuple(feature.feature_id for feature in matching),
                "blocked_by_feature_ids": blocked_by,
            })
            updated_cells.append(replace(
                cell,
                searchable=cell.searchable and not blocked_by,
                semantic_labels=tuple(dict.fromkeys(cell.semantic_labels + labels)),
                metadata=metadata,
            ))
        return replace(grid, cells=tuple(updated_cells))

    def extract_features(
        self,
        scene_nodes: Iterable[Mapping[str, Any]],
    ) -> Tuple[SemanticFeature, ...]:
        """Extract environment geometry while excluding props and robots."""
        features = []
        for node in scene_nodes:
            if not isinstance(node, Mapping):
                continue
            properties = node.get("properties") or {}
            if not isinstance(properties, Mapping):
                continue
            category = normalize_semantic_label(properties.get("category") or "")
            if category not in self.allowed_categories:
                continue
            geometry = self._shape_to_geometry(node.get("shape"))
            if geometry is None:
                continue
            feature_id = str(node.get("id") or properties.get("label") or "").strip()
            if not feature_id:
                continue
            features.append(SemanticFeature(
                feature_id=feature_id,
                category=category,
                feature_type=str(properties.get("type") or category),
                label=str(properties.get("label") or feature_id),
                geometry=geometry,
                passability=properties.get("passability"),
                metadata={
                    "description": properties.get("description"),
                    "visibility": properties.get("visibility"),
                },
            ))
        return tuple(features)

    def _shape_to_geometry(
        self,
        shape: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(shape, Mapping):
            return None
        shape_type = normalize_semantic_label(shape.get("type") or "")
        if shape_type == "polygon":
            vertices = shape.get("vertices") or ()
            return {"kind": "area", "coords": vertices} if len(vertices) >= 3 else None
        if shape_type == "rectangle":
            minimum = shape.get("min_corner") or ()
            maximum = shape.get("max_corner") or ()
            if len(minimum) < 2 or len(maximum) < 2:
                return None
            return {
                "kind": "rectangle",
                "coords": [
                    [minimum[0], minimum[1]],
                    [maximum[0], minimum[1]],
                    [maximum[0], maximum[1]],
                    [minimum[0], maximum[1]],
                ],
            }
        if shape_type == "circle":
            center = shape.get("center") or ()
            radius = shape.get("radius")
            if len(center) < 2 or radius is None:
                return None
            return {"kind": "circle", "center": center, "radius": radius}
        if shape_type == "point":
            center = shape.get("center") or ()
            if len(center) < 2:
                return None
            return {
                "kind": "point",
                "coords": [center],
                "buffer": self.point_buffer_m,
            }
        if shape_type in ("linestring", "polyline"):
            points = shape.get("points") or ()
            if len(points) < 2:
                return None
            return {
                "kind": "line",
                "coords": points,
                "buffer": self.line_buffer_m,
            }
        return None
