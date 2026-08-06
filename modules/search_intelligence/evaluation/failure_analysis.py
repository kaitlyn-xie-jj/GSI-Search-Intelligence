"""Trace-based detection failure taxonomy for frozen search experiments."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .contracts import SearchBenchmarkScenario, SearchEpisodeResult
from .success_first_experiment import _run_reference_d
from .unified_benchmark import search_skill_scenario_matrix


FAILURE_TAXONOMY = {
    "not_searched": (
        "No observation footprint contained the ground-truth target cell."
    ),
    "searched_but_occluded": (
        "The target cell entered an observation footprint, but none of those "
        "target-area observations had a valid visible line of sight."
    ),
    "visible_but_missed": (
        "At least one target-area observation was visible, but no ground-truth "
        "target detection was produced."
    ),
    "false_negative_confirmation": (
        "At least one valid target detection was produced, but the task's "
        "independent confirmation requirement was not satisfied."
    ),
    "bad_localization": (
        "A ground-truth target detection was produced, but none met the "
        "configured localization requirement."
    ),
}


def analyze_detection_failures(
    *,
    repetitions: int = 20,
    base_seed: int = 10,
) -> Mapping[str, object]:
    """Re-run the frozen D-high-res condition and classify every failure."""
    scenarios = search_skill_scenario_matrix()
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    episodes = _run_reference_d(
        scenarios,
        repetitions=repetitions,
        base_seed=base_seed,
        high_resolution_camera=True,
    )
    records = tuple(
        _classify_episode(episode, scenario_by_id[episode.scenario_id])
        for episode in episodes
    )
    total = len(records)
    successes = sum(record["success"] for record in records)
    failures = total - successes
    failure_records = tuple(record for record in records if not record["success"])
    category_counts = Counter(
        record["failure_category"] for record in failure_records
    )
    searched = sum(record["stages"]["searched"] for record in records)
    visible = sum(record["stages"]["visible"] for record in records)
    detected = sum(record["stages"]["detected"] for record in records)
    localized = sum(record["stages"]["localized"] for record in records)
    target_successes = math.ceil(0.70 * total)
    success_gap = max(0, target_successes - successes)
    return {
        "schema_version": "gsi-detection-failure-taxonomy-v1",
        "analysis_id": "D-high-res-held-out-failure-analysis-2026-08-06",
        "source_experiment": (
            "results/search_skill_success_first/experiment_report.json"
        ),
        "method": "D_high_res_camera",
        "model_changed": False,
        "seed_block": {
            "base_seed": base_seed,
            "repetitions_per_condition": repetitions,
        },
        "episode_count": total,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": successes / total,
        "taxonomy": FAILURE_TAXONOMY,
        "probability_decomposition": {
            "definition": (
                "P(success) = P(searched) * P(visible|searched) * "
                "P(detected|visible) * P(confirmed|detected)"
            ),
            "stage_counts": {
                "searched": searched,
                "visible": visible,
                "detected": detected,
                "localized": localized,
                "confirmed": successes,
            },
            "probabilities": {
                "p_searched": _ratio(searched, total),
                "p_visible_given_searched": _ratio(visible, searched),
                "p_detected_given_visible": _ratio(detected, visible),
                "p_localized_given_detected": _ratio(localized, detected),
                "p_confirmed_given_detected": _ratio(successes, detected),
                "product_to_success": (
                    _ratio(searched, total)
                    * _ratio(visible, searched)
                    * _ratio(detected, visible)
                    * _ratio(successes, detected)
                ),
            },
        },
        "failure_summary": {
            category: {
                "count": category_counts.get(category, 0),
                "fraction_of_failures": _ratio(
                    category_counts.get(category, 0),
                    failures,
                ),
                "fraction_of_all_episodes": _ratio(
                    category_counts.get(category, 0),
                    total,
                ),
            }
            for category in FAILURE_TAXONOMY
        },
        "breakdown_by_environment": _breakdown(
            records,
            key="environment",
        ),
        "breakdown_by_prior": _breakdown(records, key="prior_condition"),
        "breakdown_by_sensor_condition": _breakdown(
            records,
            key="sensor_condition",
        ),
        "path_to_70_percent": {
            "target_success_rate": 0.70,
            "target_success_count": target_successes,
            "additional_successes_required": success_gap,
            "available_failed_episodes": failures,
            "required_fraction_of_all_failures": _ratio(success_gap, failures),
            "single_category_counterfactuals": {
                category: {
                    "available_failures": category_counts.get(category, 0),
                    "success_rate_if_all_recovered": _ratio(
                        successes + category_counts.get(category, 0),
                        total,
                    ),
                    "fraction_of_category_needed_for_70_percent": (
                        _ratio(success_gap, category_counts.get(category, 0))
                        if category_counts.get(category, 0) >= success_gap
                        else None
                    ),
                }
                for category in FAILURE_TAXONOMY
            },
        },
        "failed_episodes": failure_records,
    }


def write_detection_failure_analysis(
    payload: Mapping[str, object],
    output_path: str,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _classify_episode(
    episode: SearchEpisodeResult,
    scenario: SearchBenchmarkScenario,
) -> Mapping[str, object]:
    target_attempts = tuple(
        item for item in episode.sensor_trace
        if scenario.target_cell_id in item.get("visible_cell_ids", ())
    )
    visible_attempts = tuple(
        item for item in target_attempts if bool(item.get("view_is_visible"))
    )
    target_detections = tuple(
        detection
        for item in episode.sensor_trace
        for detection in item.get("detections", ())
        if bool(detection.get("attributes", {}).get("ground_truth_match"))
    )
    maximum_error = scenario.task.success_criteria.max_localization_error_m
    localized_detections = tuple(
        detection for detection in target_detections
        if _localization_acceptable(detection, maximum_error)
    )
    searched = bool(target_attempts)
    visible = bool(visible_attempts)
    detected = bool(target_detections)
    localized = bool(localized_detections)
    category = None
    if not episode.target_found:
        if not searched:
            category = "not_searched"
        elif not visible:
            category = "searched_but_occluded"
        elif not detected:
            category = "visible_but_missed"
        elif not localized:
            category = "bad_localization"
        else:
            category = "false_negative_confirmation"
    target_observations = tuple({
        "viewpoint_key": item.get("viewpoint_key"),
        "view_is_visible": bool(item.get("view_is_visible")),
        "visibility_probability": item.get("visibility_probability"),
        "observation_quality": item.get("observation_quality"),
        "ground_truth_detection_count": sum(
            bool(detection.get("attributes", {}).get("ground_truth_match"))
            for detection in item.get("detections", ())
        ),
    } for item in target_attempts)
    return {
        "scenario_id": episode.scenario_id,
        "repetition": episode.repetition,
        "seed": episode.seed,
        "environment": scenario.metadata["environment"],
        "prior_condition": scenario.prior_condition,
        "sensor_condition": scenario.metadata["sensor_condition"],
        "success": episode.target_found,
        "failure_category": category,
        "terminal_status": episode.terminal_status,
        "stages": {
            "searched": searched,
            "visible": visible,
            "detected": detected,
            "localized": localized,
            "confirmed": episode.target_found,
        },
        "target_area_observation_count": len(target_attempts),
        "visible_target_observation_count": len(visible_attempts),
        "target_detection_count": len(target_detections),
        "valid_localized_detection_count": len(localized_detections),
        "required_confirmation_count": (
            scenario.task.success_criteria.min_confirmations
        ),
        "elapsed_time_s": episode.elapsed_time_s,
        "distance_travelled_m": episode.distance_travelled_m,
        "target_observations": target_observations,
    }


def _localization_acceptable(
    detection: Mapping[str, Any],
    maximum_error: float | None,
) -> bool:
    if maximum_error is None:
        return True
    error = detection.get("attributes", {}).get("localization_error_m")
    return error is not None and float(error) <= maximum_error


def _breakdown(records: Sequence[Mapping[str, object]], *, key: str):
    result: Dict[str, object] = {}
    for value in sorted({str(record[key]) for record in records}):
        selected = tuple(record for record in records if str(record[key]) == value)
        failures = tuple(record for record in selected if not record["success"])
        counts = Counter(record["failure_category"] for record in failures)
        result[value] = {
            "episode_count": len(selected),
            "success_count": sum(record["success"] for record in selected),
            "success_rate": _ratio(
                sum(record["success"] for record in selected),
                len(selected),
            ),
            "failure_counts": {
                category: counts.get(category, 0)
                for category in FAILURE_TAXONOMY
            },
        }
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
