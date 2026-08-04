#!/usr/bin/env python3
"""Validate the current search loop against the imported Yungu semantic map."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    BinarySensorModel,
    SearchBenchmarkConfig,
    SearchBenchmarkRunner,
    SearchBenchmarkScenario,
    SearchGrid,
    SearchPrior,
    SearchTask,
    SemanticGridBuilder,
    compare_paired_policy_results,
    write_benchmark_report,
)


TARGET_LABELS = (
    "parking",
    "road",
    "plaza",
    "passage",
    "building_frontage",
    "access_point",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a paired active-versus-lookahead feasibility benchmark on Yungu."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "yungu2030_v1",
    )
    parser.add_argument("--resolution-m", type=float, default=10.0)
    parser.add_argument("--altitude-m", type=float, default=70.0)
    parser.add_argument("--footprint-radius-m", type=float, default=35.0)
    parser.add_argument("--candidate-stride-cells", type=int, default=3)
    parser.add_argument("--max-viewpoints", type=int, default=18)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "yungu_map_feasibility",
    )
    args = parser.parse_args()

    semantic_path = args.asset_root / "semantic_map.json"
    sdf_path = args.asset_root / "yungu_local_origin.sdf"
    semantic_map = _read_json(semantic_path)
    _validate_map_identity(semantic_map, sdf_path)

    task = SearchTask.from_skill_params({
        "task_id": "yungu-map-feasibility",
        "area_token": semantic_map["world_name"],
        "area": semantic_map["search_area"],
        "target_token": "yellow-van",
        "max_viewpoints": args.max_viewpoints,
    })
    grid = SemanticGridBuilder().annotate(
        SearchGrid.from_task(task, args.resolution_m),
        semantic_map["nodes"],
    )
    scenarios = _scenarios(task, grid)
    config = SearchBenchmarkConfig(
        policy_names=("active", "lookahead_active"),
        repetitions=args.repetitions,
        base_seed=args.seed,
        altitude_m=args.altitude_m,
        footprint_radius_m=args.footprint_radius_m,
        candidate_stride_cells=args.candidate_stride_cells,
        max_candidates=None,
        distance_scale_mode="map_diagonal",
        sensor_model=BinarySensorModel(0.85, 0.01),
    )
    report = SearchBenchmarkRunner(config).run(scenarios)
    artifacts = write_benchmark_report(report, args.output_dir)
    episodes = report.episodes
    comparisons = compare_paired_policy_results(
        [item for item in episodes if item.policy_name == "active"],
        [item for item in episodes if item.policy_name == "lookahead_active"],
        scenarios,
    )
    manifest = {
        "schema_version": "gsi-yungu-feasibility-v1",
        "scope": (
            "Offline semantic-map integration validation only; this is not a "
            "Gazebo, visibility, collision, or flight-dynamics experiment."
        ),
        "asset_root": str(args.asset_root.resolve()),
        "semantic_map": str(semantic_path.resolve()),
        "sdf": str(sdf_path.resolve()),
        "world_name": semantic_map["world_name"],
        "grid": {
            "resolution_m": grid.resolution_m,
            "cell_count": len(grid.cells),
            "searchable_cell_count": len(grid.searchable_cells),
        },
        "scenarios": [item.to_dict() for item in scenarios],
        "config": config.to_dict(),
        "paired_comparisons": {
            name: comparison.to_dict()
            for name, comparison in comparisons.items()
        },
        "artifacts": artifacts,
    }
    manifest_path = args.output_dir / "yungu_feasibility_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "artifact_manifest": str(manifest_path),
        "paired_missions": len(scenarios) * args.repetitions,
        "policy_episodes": len(episodes),
        "searchable_cells": len(grid.searchable_cells),
        "comparisons": {
            name: comparison.to_dict()
            for name, comparison in comparisons.items()
        },
    }, indent=2, sort_keys=True))
    return 0


def _scenarios(
    task: SearchTask,
    grid: SearchGrid,
) -> tuple[SearchBenchmarkScenario, ...]:
    start = grid.nearest_searchable_cell(160.0, 110.0)
    if start is None:
        raise ValueError("Yungu map has no searchable start cell")
    scenarios = []
    for label in TARGET_LABELS:
        matching = sorted(
            (cell for cell in grid.searchable_cells if label in cell.semantic_labels),
            key=lambda cell: cell.cell_id,
        )
        if not matching:
            raise ValueError(f"Yungu map has no searchable cells labelled {label}")
        target = matching[len(matching) // 2]
        prior = SearchPrior(
            task_id=f"yungu-{label}",
            semantic_weights={label: 1.0},
            confidence=0.85,
            default_weight=0.05,
            projection_mode="label_mass",
        ).project(grid)
        scenarios.append(SearchBenchmarkScenario(
            scenario_id=f"yungu-{label}",
            task=SearchTask.from_skill_params({
                "task_id": f"yungu-{label}",
                "area_token": task.search_area.area_id,
                "area": task.search_area.geometry,
                "target_token": task.target.query,
                "max_viewpoints": task.budget.max_viewpoints,
            }),
            grid=grid,
            target_cell_id=target.cell_id,
            initial_belief=prior.belief,
            start_xy=start.center,
            prior_condition="semantic_correct",
            metadata={
                "map_id": "yungu2030_local_origin",
                "target_semantic_label": label,
                "target_feature_count": len(matching),
                "prior_confidence": prior.confidence,
            },
        ))
    return tuple(scenarios)


def _validate_map_identity(
    semantic_map: Mapping[str, Any],
    sdf_path: Path,
) -> None:
    if not sdf_path.is_file():
        raise FileNotFoundError(f"Yungu SDF is missing: {sdf_path}")
    root = ET.parse(sdf_path).getroot()
    world = root.find("world")
    if world is None:
        raise ValueError("Yungu SDF has no world element")
    if semantic_map.get("world_name") != world.get("name"):
        raise ValueError("semantic_map world_name does not match the SDF world name")
    semantic_buildings = {
        str(node["id"])
        for node in semantic_map.get("nodes", ())
        if node.get("properties", {}).get("category") == "building"
    }
    collision_buildings = {
        collision.get("name")
        for model in world.findall("model")
        if model.get("name") == "yungu_campus_local_origin"
        for collision in model.findall(".//collision")
        if collision.find("geometry/box") is not None
    }
    if semantic_buildings != collision_buildings:
        raise ValueError("SDF building collisions and semantic building IDs differ")


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Yungu semantic map is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, Mapping):
        raise TypeError("Yungu semantic map must be a JSON object")
    return document


if __name__ == "__main__":
    raise SystemExit(main())
