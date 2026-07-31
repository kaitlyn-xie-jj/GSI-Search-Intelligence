"""Summarize repeatable SearchWorld V2 Gazebo/PX4 trials."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SUMMARY_FIELDS = (
    "scenario",
    "seed",
    "status",
    "found",
    "steps",
    "observation_count",
    "command_count",
    "elapsed_time_s",
    "distance_travelled_m",
    "confidence",
    "coverage_fraction",
    "belief_entropy_nats",
    "localization_error_m",
    "trace_path",
)


def load_trace(path: Path | str) -> list[Mapping[str, Any]]:
    events = []
    trace_path = Path(path)
    if not trace_path.is_file():
        return events
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"trace line {line_number} is not a JSON object")
        events.append(value)
    return events


def summarize_trial(
    scenario: str,
    trace_path: Path | str,
    ground_truth_path: Path | str,
) -> Dict[str, Any]:
    events = load_trace(trace_path)
    truth = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    outcomes = [event["outcome"] for event in events if event.get("event") == "outcome"]
    outcome = outcomes[-1] if outcomes else {}
    metrics = outcome.get("metrics") or {}
    target = truth["targets"][0]
    estimate = outcome.get("estimated_target_position")
    localization_error = None
    if isinstance(estimate, list) and len(estimate) >= 2:
        pose = target["pose_enu_m"]
        localization_error = math.hypot(float(estimate[0]) - pose["x"], float(estimate[1]) - pose["y"])
    return {
        "schema_version": "gsi-searchworld-v2-trial-v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scenario": scenario,
        "seed": truth.get("seed"),
        "status": outcome.get("status", "no_outcome"),
        "found": bool(outcome.get("found", False)),
        "steps": outcome.get("steps"),
        "observation_count": sum(event.get("event") == "observation" for event in events),
        "command_count": sum(event.get("event") == "command" for event in events),
        "elapsed_time_s": outcome.get("elapsed_time_s"),
        "distance_travelled_m": outcome.get("distance_travelled_m"),
        "confidence": outcome.get("confidence"),
        "coverage_fraction": metrics.get("coverage_fraction"),
        "belief_entropy_nats": metrics.get("belief_entropy_nats"),
        "localization_error_m": localization_error,
        "target_slot_id": target.get("slot_id"),
        "target_semantic_region_id": target.get("semantic_region_id"),
        "trace_path": str(Path(trace_path)),
    }


def write_summary(summary: Mapping[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate_summaries(summaries: Iterable[Mapping[str, Any]], output_dir: Path | str) -> Dict[str, Any]:
    rows = list(summaries)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    found = [row for row in rows if row.get("found")]
    aggregate = {
        "schema_version": "gsi-searchworld-v2-batch-v1",
        "trial_count": len(rows),
        "success_count": len(found),
        "success_rate": len(found) / len(rows) if rows else None,
        "scenarios": sorted({str(row.get("scenario")) for row in rows}),
        "mean_elapsed_time_s_success": _mean(row.get("elapsed_time_s") for row in found),
        "mean_distance_m_success": _mean(row.get("distance_travelled_m") for row in found),
        "mean_localization_error_m_success": _mean(row.get("localization_error_m") for row in found),
        "trials": rows,
    }
    write_summary(aggregate, output / "batch_summary.json")
    return aggregate


def _mean(values: Iterable[Any]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aggregate-root", type=Path)
    options = parser.parse_args(args)
    if options.aggregate_root:
        paths = sorted(options.aggregate_root.glob("*/summary.json"))
        summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        aggregate_summaries(summaries, options.aggregate_root)
        return
    if not all((options.scenario, options.trace, options.ground_truth, options.output)):
        parser.error("single-trial mode requires --scenario, --trace, --ground-truth, and --output")
    write_summary(
        summarize_trial(options.scenario, options.trace, options.ground_truth),
        options.output,
    )


if __name__ == "__main__":
    main()
