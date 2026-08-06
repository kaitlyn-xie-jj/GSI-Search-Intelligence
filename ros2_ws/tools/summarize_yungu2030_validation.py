#!/usr/bin/env python3
"""Validate, classify, and aggregate Yungu2030 Gazebo/PX4 episodes."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WILSON_Z_95 = 1.959963984540054
TRIAL_FIELDS = (
    "episode_id",
    "batch_mode",
    "cohort",
    "target_id",
    "semantic_region",
    "repetition",
    "started_utc",
    "ended_utc",
    "runner_status",
    "search_exit_status",
    "status",
    "success",
    "failure_category",
    "outcome_status",
    "observation_count",
    "observation_triggers",
    "observation_qualities",
    "confidence",
    "localization_error_m",
    "elapsed_time_s",
    "distance_travelled_m",
    "artifact_complete",
    "artifact_bytes",
    "trial_dir",
)


class ManifestValidationError(ValueError):
    """Raised when a frozen target does not match the semantic map."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _inside_rectangle(x: float, y: float, shape: Mapping[str, Any]) -> bool:
    minimum = shape.get("min_corner")
    maximum = shape.get("max_corner")
    return (
        shape.get("type") == "rectangle"
        and isinstance(minimum, list)
        and isinstance(maximum, list)
        and len(minimum) >= 2
        and len(maximum) >= 2
        and float(minimum[0]) <= x <= float(maximum[0])
        and float(minimum[1]) <= y <= float(maximum[1])
    )


def manifest_targets(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    stability = dict(manifest["stability"]["target"])
    positions = [dict(target) for target in manifest["positions"]["targets"]]
    return [stability, *positions]


def validate_manifest(
    manifest_path: Path | str,
    semantic_map_path: Path | str | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    manifest = _read_json(manifest_file)
    if semantic_map_path is None:
        semantic_map_path = manifest_file.parent / manifest["semantic_map"]
    semantic_file = Path(semantic_map_path).resolve()
    semantic_map = _read_json(semantic_file)
    if semantic_map.get("world_name") != manifest.get("world_name"):
        raise ManifestValidationError("manifest and semantic map world names differ")

    nodes = semantic_map.get("nodes") or []
    node_by_id = {node.get("id"): node for node in nodes}
    restricted = [
        node
        for node in nodes
        if (node.get("properties") or {}).get("passability") == "restricted"
    ]
    search_shape = {
        "type": semantic_map["search_area"].get("kind"),
        "min_corner": [
            min(point[0] for point in semantic_map["search_area"]["coords"]),
            min(point[1] for point in semantic_map["search_area"]["coords"]),
        ],
        "max_corner": [
            max(point[0] for point in semantic_map["search_area"]["coords"]),
            max(point[1] for point in semantic_map["search_area"]["coords"]),
        ],
    }
    identifiers: set[str] = set()
    validated: list[dict[str, Any]] = []
    for target in manifest_targets(manifest):
        target_id = str(target["target_id"])
        if target_id in identifiers:
            raise ManifestValidationError(f"duplicate target_id: {target_id}")
        identifiers.add(target_id)
        x, y = float(target["x"]), float(target["y"])
        region_id = str(target["semantic_region"])
        region = node_by_id.get(region_id)
        if region is None:
            raise ManifestValidationError(
                f"{target_id}: semantic region {region_id} does not exist"
            )
        properties = region.get("properties") or {}
        if properties.get("passability") != "open":
            raise ManifestValidationError(f"{target_id}: region {region_id} is not open")
        if not _inside_rectangle(x, y, region.get("shape") or {}):
            raise ManifestValidationError(
                f"{target_id}: ({x}, {y}) is outside {region_id}"
            )
        if not _inside_rectangle(x, y, search_shape):
            raise ManifestValidationError(f"{target_id}: position is outside search area")
        collisions = [
            str(node.get("id"))
            for node in restricted
            if _inside_rectangle(x, y, node.get("shape") or {})
        ]
        if collisions:
            raise ManifestValidationError(
                f"{target_id}: position intersects restricted nodes {collisions}"
            )
        validated.append(target)
    return {
        "manifest": str(manifest_file),
        "semantic_map": str(semantic_file),
        "sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "target_count": len(validated),
        "valid": True,
    }


def expand_episodes(
    manifest: Mapping[str, Any], mode: str
) -> list[dict[str, Any]]:
    stability_target = dict(manifest["stability"]["target"])
    position_targets = {
        target["target_id"]: dict(target)
        for target in manifest["positions"]["targets"]
    }
    episodes: list[dict[str, Any]] = []

    def add(cohort: str, target: Mapping[str, Any], repetition: int) -> None:
        prefix = "preflight__" if mode == "preflight" else ""
        episodes.append(
            {
                "episode_id": (
                    f"{prefix}{cohort}__{target['target_id']}__r{repetition:02d}"
                ),
                "cohort": cohort,
                "repetition": repetition,
                **dict(target),
            }
        )

    if mode == "preflight":
        for item in manifest["preflight"]:
            cohort = item["cohort"]
            target = (
                stability_target
                if cohort == "stability"
                else position_targets[item["target_id"]]
            )
            add(cohort, target, int(item["repetition"]))
    if mode in ("stability", "all"):
        for repetition in range(1, int(manifest["stability"]["repetitions"]) + 1):
            add("stability", stability_target, repetition)
    if mode in ("positions", "all"):
        repetitions = int(manifest["positions"]["repetitions_per_target"])
        for target in manifest["positions"]["targets"]:
            for repetition in range(1, repetitions + 1):
                add("positions", target, repetition)
    return episodes


def wilson_interval(successes: int, total: int) -> dict[str, Any]:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("successes must satisfy 0 <= successes <= total")
    if total == 0:
        return {
            "confidence_level": 0.95,
            "lower": None,
            "upper": None,
        }
    proportion = successes / total
    denominator = 1.0 + WILSON_Z_95 * WILSON_Z_95 / total
    center = (
        proportion + WILSON_Z_95 * WILSON_Z_95 / (2.0 * total)
    ) / denominator
    margin = WILSON_Z_95 / denominator * math.sqrt(
        proportion * (1.0 - proportion) / total
        + WILSON_Z_95 * WILSON_Z_95 / (4.0 * total * total)
    )
    return {
        "confidence_level": 0.95,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _load_trace(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events, "trace_missing"
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {number} is not an object")
            events.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        return events, f"trace_invalid: {error}"
    return events, None


def _artifact_status(trial_dir: Path) -> tuple[dict[str, bool], int]:
    capture = trial_dir / "capture"
    candidates = {
        "rgb": [capture / "rgb.mp4", capture / "rgb.rgb24"],
        "depth": [capture / "sensor_snapshot" / "depth_image.raw"],
        "point_cloud": [capture / "sensor_snapshot" / "point_cloud.bin"],
        "rosbag": [*capture.glob("rosbag2/*.db3"), *capture.glob("rosbag2/*.mcap")],
        "trace": [trial_dir / "search_trace.jsonl"],
    }
    status = {
        name: any(path.is_file() and path.stat().st_size > 0 for path in paths)
        for name, paths in candidates.items()
    }
    artifact_bytes = sum(
        path.stat().st_size
        for path in trial_dir.rglob("*")
        if path.is_file() and path.name != "trial_summary.json"
    )
    return status, artifact_bytes


def _read_exit_status(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _resolve_trial_manifest(trial_root: Path, recorded_path: str) -> Path:
    """Resolve metadata paths after artifacts move between WSL and Windows."""
    recorded = Path(recorded_path)
    if recorded.is_file():
        return recorded
    batch_local = trial_root.parents[1] / recorded.name
    if batch_local.is_file():
        return batch_local
    raise FileNotFoundError(recorded_path)


def _confirmation_observations(
    observations: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return observations that contributed positive target evidence."""
    return [item for item in observations if item.get("detections")]


def classify_trial(
    *,
    outcome: Mapping[str, Any],
    observation_qualities: Sequence[float],
    observation_triggers: Sequence[str | None],
    localization_error_m: float | None,
    artifact_complete: bool,
    trace_error: str | None,
    runner_status: int,
    search_exit_status: int | None,
    minimum_observations: int,
    minimum_quality: float,
    required_second_trigger: str,
    maximum_localization_error_m: float,
) -> tuple[bool, str | None]:
    if search_exit_status == 124 or (runner_status == 124 and not outcome):
        return False, "search_timeout"
    if trace_error or not artifact_complete:
        return False, "infrastructure_artifact_failure"
    if not outcome:
        return False, "infrastructure_artifact_failure"
    if not bool(outcome.get("found")) or outcome.get("status") != "found":
        return False, "algorithm_failure"
    if len(observation_qualities) < minimum_observations:
        return False, "observation_independence_failure"
    if (
        len(observation_triggers) < 2
        or observation_triggers[1] != required_second_trigger
    ):
        return False, "observation_independence_failure"
    if any(quality < minimum_quality for quality in observation_qualities):
        return False, "observation_quality_failure"
    if (
        localization_error_m is None
        or localization_error_m > maximum_localization_error_m
    ):
        return False, "localization_failure"
    if runner_status != 0:
        return False, "infrastructure_artifact_failure"
    return True, None


def summarize_trial(trial_dir: Path | str) -> dict[str, Any]:
    root = Path(trial_dir).resolve()
    metadata = _read_json(root / "trial_metadata.json")
    manifest = _read_json(_resolve_trial_manifest(
        root,
        str(metadata["manifest_path"]),
    ))
    config = manifest["configuration"]
    events, trace_error = _load_trace(root / "search_trace.jsonl")
    outcomes = [event.get("outcome") for event in events if event.get("event") == "outcome"]
    outcome = dict(outcomes[-1] or {}) if outcomes else {}
    observations = [
        event.get("observation") or {}
        for event in events
        if event.get("event") == "observation"
    ]
    qualities = [float(item.get("observation_quality", 0.0)) for item in observations]
    triggers = [
        (item.get("sensor_metadata") or {}).get("observation_trigger")
        for item in observations
    ]
    confirmations = _confirmation_observations(observations)
    confirmation_qualities = [
        float(item.get("observation_quality", 0.0))
        for item in confirmations
    ]
    confirmation_triggers = [
        (item.get("sensor_metadata") or {}).get("observation_trigger")
        for item in confirmations
    ]
    estimate = outcome.get("estimated_target_position")
    target = metadata["target"]
    localization_error = None
    if isinstance(estimate, list) and len(estimate) >= 2:
        localization_error = math.hypot(
            float(estimate[0]) - float(target["x"]),
            float(estimate[1]) - float(target["y"]),
        )
    artifacts, artifact_bytes = _artifact_status(root)
    artifact_complete = all(artifacts.values())
    runner_status = int(metadata.get("runner_status", 1))
    search_exit_status = _read_exit_status(root / "search_exit_status.txt")
    success, failure_category = classify_trial(
        outcome=outcome,
        observation_qualities=confirmation_qualities,
        observation_triggers=confirmation_triggers,
        localization_error_m=localization_error,
        artifact_complete=artifact_complete,
        trace_error=trace_error,
        runner_status=runner_status,
        search_exit_status=search_exit_status,
        minimum_observations=int(config["minimum_observations"]),
        minimum_quality=float(config["minimum_observation_quality"]),
        required_second_trigger=str(config["required_second_observation_trigger"]),
        maximum_localization_error_m=float(
            config["maximum_horizontal_localization_error_m"]
        ),
    )
    summary = {
        "schema_version": "gsi-yungu2030-validation-trial-v1",
        **metadata,
        "status": "success" if success else "failure",
        "success": success,
        "failure_category": failure_category,
        "search_exit_status": search_exit_status,
        "outcome_status": outcome.get("status", "no_outcome"),
        "outcome_reason": outcome.get("reason"),
        "observation_count": len(observations),
        "observation_triggers": triggers,
        "observation_qualities": qualities,
        "confirmation_observation_count": len(confirmations),
        "confirmation_observation_triggers": confirmation_triggers,
        "confirmation_observation_qualities": confirmation_qualities,
        "confidence": outcome.get("confidence"),
        "estimated_target_position": estimate,
        "localization_error_m": localization_error,
        "elapsed_time_s": outcome.get("elapsed_time_s"),
        "distance_travelled_m": outcome.get("distance_travelled_m"),
        "artifact_complete": artifact_complete,
        "artifacts": artifacts,
        "artifact_bytes": artifact_bytes,
        "trace_error": trace_error,
        "trial_dir": str(root),
    }
    _write_json_atomic(root / "trial_summary.json", summary)
    return summary


def _mean(values: Iterable[Any]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _rate(rows: Sequence[Mapping[str, Any]], field: str = "success") -> dict[str, Any]:
    successes = sum(bool(row.get(field)) for row in rows)
    return {
        "trial_count": len(rows),
        "success_count": successes,
        "success_rate": successes / len(rows) if rows else None,
        "wilson_95": wilson_interval(successes, len(rows)),
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("success")]
    return {
        **_rate(rows),
        "artifact_completeness": _rate(rows, "artifact_complete"),
        "failure_categories": dict(
            sorted(Counter(row.get("failure_category") for row in rows if row.get("failure_category")).items())
        ),
        "mean_elapsed_time_s_success": _mean(
            row.get("elapsed_time_s") for row in successful
        ),
        "mean_distance_travelled_m_success": _mean(
            row.get("distance_travelled_m") for row in successful
        ),
        "mean_localization_error_m_success": _mean(
            row.get("localization_error_m") for row in successful
        ),
    }


def aggregate_batch(batch_root: Path | str) -> dict[str, Any]:
    root = Path(batch_root).resolve()
    metadata = _read_json(root / "batch_metadata.json")
    manifest = _read_json(root / "validation_manifest.json")
    paths = sorted((root / "episodes").glob("*/trial_summary.json"))
    rows = [_read_json(path) for path in paths]
    csv_path = root / "trials.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["observation_triggers"] = json.dumps(row.get("observation_triggers"))
            flat["observation_qualities"] = json.dumps(row.get("observation_qualities"))
            writer.writerow(flat)
    os.replace(csv_temporary, csv_path)

    cohorts = {
        cohort: _group_summary([row for row in rows if row.get("cohort") == cohort])
        for cohort in ("stability", "positions")
    }
    targets = {
        target_id: {
            "cohort": target_rows[0].get("cohort"),
            "semantic_region": target_rows[0].get("semantic_region"),
            **_group_summary(target_rows),
        }
        for target_id in sorted({str(row.get("target_id")) for row in rows})
        if (target_rows := [row for row in rows if str(row.get("target_id")) == target_id])
    }
    position_targets = [
        value
        for value in targets.values()
        if value["cohort"] == "positions"
    ]
    position_target_successes = sum(value["success_count"] > 0 for value in position_targets)
    target_coverage = {
        "target_count": len(position_targets),
        "targets_with_success": position_target_successes,
        "rate": (
            position_target_successes / len(position_targets)
            if position_targets
            else None
        ),
        "wilson_95": wilson_interval(position_target_successes, len(position_targets)),
    }
    acceptance = manifest["acceptance"]
    expected = int(metadata["expected_episode_count"])
    summary = {
        "schema_version": "gsi-yungu2030-validation-batch-v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "batch_mode": metadata["batch_mode"],
        "expected_episode_count": expected,
        "completed_episode_count": len(rows),
        "complete": len(rows) == expected,
        "overall": _group_summary(rows),
        "cohorts": cohorts,
        "targets": targets,
        "position_target_coverage": target_coverage,
        "acceptance": {
            "artifact_completeness_pass": bool(rows)
            and _rate(rows, "artifact_complete")["success_rate"]
            >= float(acceptance["artifact_minimum_completeness_rate"]),
            "stability_success_pass": bool(cohorts["stability"]["trial_count"])
            and cohorts["stability"]["success_rate"]
            >= float(acceptance["stability_minimum_success_rate"]),
            "positions_success_pass": bool(cohorts["positions"]["trial_count"])
            and cohorts["positions"]["success_rate"]
            >= float(acceptance["positions_minimum_success_rate"]),
            "positions_target_coverage_pass": target_coverage["rate"] is not None
            and target_coverage["rate"]
            >= float(acceptance["positions_minimum_targets_with_success_rate"]),
        },
        "manifest_sha256": metadata["manifest_sha256"],
    }
    _write_json_atomic(root / "batch_summary.json", summary)
    return summary


def write_trial_metadata(options: argparse.Namespace) -> None:
    path = options.trial_dir.resolve() / "trial_metadata.json"
    existing = _read_json(path) if path.is_file() else {}
    value = {
        **existing,
        "episode_id": options.episode_id,
        "batch_mode": options.batch_mode,
        "cohort": options.cohort,
        "target_id": options.target_id,
        "semantic_region": options.semantic_region,
        "repetition": options.repetition,
        "target": {
            "x": options.x,
            "y": options.y,
            "z": options.z,
            "yaw": options.yaw,
        },
        "manifest_path": str(options.manifest.resolve()),
        "started_utc": options.started_utc,
        "ended_utc": options.ended_utc,
        "runner_status": options.runner_status,
    }
    _write_json_atomic(path, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--semantic-map", type=Path)

    episodes = subparsers.add_parser("episodes")
    episodes.add_argument("--manifest", required=True, type=Path)
    episodes.add_argument(
        "--mode", required=True, choices=("preflight", "stability", "positions", "all")
    )
    episodes.add_argument("--target-id", action="append", default=[])

    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("--trial-dir", required=True, type=Path)
    metadata.add_argument("--manifest", required=True, type=Path)
    metadata.add_argument("--episode-id", required=True)
    metadata.add_argument("--batch-mode", required=True)
    metadata.add_argument("--cohort", required=True)
    metadata.add_argument("--target-id", required=True)
    metadata.add_argument("--semantic-region", required=True)
    metadata.add_argument("--repetition", required=True, type=int)
    metadata.add_argument("--x", required=True, type=float)
    metadata.add_argument("--y", required=True, type=float)
    metadata.add_argument("--z", required=True, type=float)
    metadata.add_argument("--yaw", required=True, type=float)
    metadata.add_argument("--started-utc", required=True)
    metadata.add_argument("--ended-utc")
    metadata.add_argument("--runner-status", type=int)

    trial = subparsers.add_parser("trial")
    trial.add_argument("--trial-dir", required=True, type=Path)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--batch-root", required=True, type=Path)
    return parser


def main(args: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(args)
    if options.command == "validate-manifest":
        print(json.dumps(validate_manifest(options.manifest, options.semantic_map)))
    elif options.command == "episodes":
        manifest = _read_json(options.manifest)
        expanded = expand_episodes(manifest, options.mode)
        if options.target_id:
            requested = set(options.target_id)
            known = {episode["target_id"] for episode in expanded}
            unknown = sorted(requested - known)
            if unknown:
                raise ValueError(f"unknown target filters: {unknown}")
            expanded = [
                episode for episode in expanded
                if episode["target_id"] in requested
            ]
        for episode in expanded:
            print("\t".join(str(episode[field]) for field in (
                "episode_id", "cohort", "target_id", "semantic_region",
                "repetition", "x", "y", "z", "yaw",
            )))
    elif options.command == "metadata":
        write_trial_metadata(options)
    elif options.command == "trial":
        print(json.dumps(summarize_trial(options.trial_dir)))
    elif options.command == "batch":
        print(json.dumps(aggregate_batch(options.batch_root)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"validation summary error: {error}", file=sys.stderr)
        raise SystemExit(2)
