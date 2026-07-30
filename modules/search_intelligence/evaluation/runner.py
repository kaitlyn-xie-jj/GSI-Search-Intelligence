"""Closed-loop episode and suite runners for policy comparison."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ..belief import BayesianBeliefUpdater, BeliefMap
from ..contracts import SearchObservation, TargetDetection, Viewpoint
from ..execution import SearchSession
from ..policies import (
    ActiveSearchPolicy,
    CoveragePolicy,
    GreedyPriorPolicy,
    RandomPolicy,
    SearchPolicy,
)
from ..search_space import CandidateViewpointGenerator, ViewpointCandidate
from .contracts import (
    SearchBenchmarkConfig,
    SearchBenchmarkReport,
    SearchBenchmarkScenario,
    SearchEpisodeResult,
)
from .reporting import aggregate_policy_results


class SearchEpisodeRunner:
    """Run all policies through the same platform-neutral observation loop."""

    def __init__(self, config: SearchBenchmarkConfig) -> None:
        self.config = config

    def run(
        self,
        scenario: SearchBenchmarkScenario,
        policy_name: str,
        repetition: int = 0,
    ) -> SearchEpisodeResult:
        policy_name = policy_name.strip().lower()
        seed = _stable_seed(
            self.config.base_seed,
            scenario.scenario_id,
            repetition,
        )
        candidates = self._candidates(scenario)
        policy = self._policy(policy_name, scenario, candidates, seed)
        initial = BeliefMap.for_grid(scenario.grid, scenario.initial_belief)
        session = SearchSession(
            scenario.task,
            policy,
            initial_belief=initial.probabilities,
            current_viewpoint=self.config.start_viewpoint(scenario),
            initial_policy_metadata={
                "search_policy": policy_name,
                "benchmark_scenario_id": scenario.scenario_id,
                "prior_condition": scenario.prior_condition,
            },
            search_grid=scenario.grid,
            belief_updater=BayesianBeliefUpdater(self.config.sensor_model),
        )
        entropy_trace = [initial.entropy_nats]
        sensor_trace = []

        while not session.completed:
            viewpoint = session.next_viewpoint()
            if viewpoint is None:
                break
            distance = _viewpoint_distance(session.state.current_viewpoint, viewpoint)
            travel_time = distance / self.config.speed_mps
            elapsed = travel_time + self.config.observation_time_s
            energy = (
                distance * self.config.energy_per_m
                + self.config.observation_energy
            )
            visible_cells = scenario.grid.cells_within_radius(
                viewpoint.x,
                viewpoint.y,
                self.config.footprint_radius_m,
            )
            visible_ids = tuple(cell.cell_id for cell in visible_cells)
            detections = self._detections(
                scenario,
                viewpoint,
                visible_ids,
                seed,
            )
            sensor_trace.append({
                "viewpoint_key": viewpoint.key,
                "visible_cell_ids": visible_ids,
                "detections": tuple(_detection_trace(item) for item in detections),
            })
            timestamp = session.state.elapsed_time_s + elapsed
            session.record_observation(SearchObservation(
                viewpoint=viewpoint,
                timestamp_s=timestamp,
                detections=detections,
                visible_cell_ids=visible_ids,
                observation_quality=self.config.observation_quality,
                travel_time_s=elapsed,
                travel_distance_m=distance,
                energy_used=energy,
                sensor_metadata={
                    "source": "search_policy_benchmark",
                    "scenario_id": scenario.scenario_id,
                    "repetition": repetition,
                    "footprint_radius_m": self.config.footprint_radius_m,
                    "observation_quality": self.config.observation_quality,
                    "effective_detection_probability": (
                        self.config.sensor_model.effective_detection_probability(
                            self.config.observation_quality
                        )
                    ),
                    "persistent_distractor_probability": (
                        self.config.persistent_distractor_probability
                    ),
                    "false_alarm_correlation": self.config.false_alarm_correlation,
                    "localization_error_std_m": self.config.localization_error_std_m,
                },
            ))
            entropy_trace.append(BeliefMap.from_mapping(
                session.state.belief
            ).entropy_nats)

        if not session.completed:
            session.next_viewpoint()
        assert session.outcome is not None
        outcome = session.outcome
        target_found = any(
            detection.entity_id == scenario.target_entity_id
            for detection in outcome.detections
        )
        declared_found = outcome.found
        belief_ids = set(session.state.belief)
        observed_ids = set(session.state.observed_cell_quality)
        coverage = (
            len(belief_ids & observed_ids) / len(belief_ids)
            if belief_ids else 0.0
        )
        shortest_distance = self._shortest_detection_distance(
            scenario,
            candidates,
        )
        spl = _spl(
            target_found,
            shortest_distance,
            session.state.distance_travelled_m,
        )
        final_entropy = BeliefMap.from_mapping(session.state.belief).entropy_nats
        all_detections = tuple(
            detection
            for observation in session.state.observations
            for detection in observation.detections
        )
        source_counts = {
            source: sum(
                detection.attributes.get("source_kind") == source
                for detection in all_detections
            )
            for source in (
                "target",
                "persistent_distractor",
                "correlated_false_alarm",
                "independent_false_alarm",
            )
        }
        localization_errors = tuple(
            float(detection.attributes["localization_error_m"])
            for detection in all_detections
            if "localization_error_m" in detection.attributes
        )
        return SearchEpisodeResult(
            scenario_id=scenario.scenario_id,
            prior_condition=scenario.prior_condition,
            policy_name=policy_name,
            repetition=repetition,
            seed=seed,
            terminal_status=outcome.status.value,
            declared_found=declared_found,
            target_found=target_found,
            false_positive=declared_found and not target_found,
            steps=session.state.step_index,
            elapsed_time_s=session.state.elapsed_time_s,
            distance_travelled_m=session.state.distance_travelled_m,
            energy_used=session.state.energy_used,
            coverage_fraction=coverage,
            spl=spl,
            shortest_detection_distance_m=shortest_distance,
            initial_entropy_nats=initial.entropy_nats,
            final_entropy_nats=final_entropy,
            entropy_reduction_nats=initial.entropy_nats - final_entropy,
            detection_count=len(all_detections),
            target_detection_count=source_counts["target"],
            persistent_distractor_count=source_counts["persistent_distractor"],
            correlated_false_alarm_count=source_counts["correlated_false_alarm"],
            independent_false_alarm_count=source_counts["independent_false_alarm"],
            mean_localization_error_m=(
                sum(localization_errors) / len(localization_errors)
                if localization_errors else 0.0
            ),
            policy_trace=session.policy_decisions,
            belief_entropy_trace=tuple(entropy_trace),
            sensor_trace=tuple(sensor_trace),
        )

    def _candidates(
        self,
        scenario: SearchBenchmarkScenario,
    ) -> Tuple[ViewpointCandidate, ...]:
        return CandidateViewpointGenerator(
            altitude_m=self.config.altitude_m,
            footprint_radius_m=self.config.footprint_radius_m,
            stride_cells=self.config.candidate_stride_cells,
            max_candidates=self.config.max_candidates,
        ).generate(scenario.grid)

    def _policy(
        self,
        policy_name: str,
        scenario: SearchBenchmarkScenario,
        candidates: Tuple[ViewpointCandidate, ...],
        seed: int,
    ) -> SearchPolicy:
        if policy_name == "coverage":
            return CoveragePolicy(
                pass_spacing_m=self.config.coverage_pass_spacing_m,
                altitude_m=self.config.altitude_m,
                observation_spacing_m=self.config.coverage_observation_spacing_m,
            )
        if policy_name == "random":
            return RandomPolicy(candidates, seed=seed)
        if policy_name == "greedy_prior":
            return GreedyPriorPolicy(
                candidates,
                scenario.initial_belief,
                distance_scale_m=self.config.distance_scale_m,
            )
        if policy_name == "active":
            return ActiveSearchPolicy(
                candidates,
                sensor_model=self.config.sensor_model,
                observation_quality=self.config.observation_quality,
                detection_weight=self.config.detection_weight,
                information_gain_weight=self.config.information_gain_weight,
                novelty_weight=self.config.novelty_weight,
                travel_weight=self.config.travel_weight,
                distance_scale_m=self.config.distance_scale_m,
                verification_followup_limit=(
                    self.config.verification_followup_limit
                ),
            )
        raise ValueError(f"unsupported benchmark policy: {policy_name}")

    def _detections(
        self,
        scenario: SearchBenchmarkScenario,
        viewpoint: Viewpoint,
        visible_cell_ids: Tuple[str, ...],
        seed: int,
    ) -> Tuple[TargetDetection, ...]:
        visible_ids = set(visible_cell_ids)
        target_visible = scenario.target_cell_id in visible_ids
        detections = []
        if target_visible:
            probability = self.config.sensor_model.effective_detection_probability(
                self.config.observation_quality
            )
            if _stable_unit_interval(
                seed,
                scenario.scenario_id,
                viewpoint.key,
                "sensor",
            ) < probability:
                detections.append(self._localized_detection(
                    scenario,
                    viewpoint,
                    scenario.target_cell.center,
                    scenario.target_entity_id,
                    "target",
                    True,
                    seed,
                ))

        distractor_cell_id = scenario.metadata.get("distractor_cell_id")
        distractor_cell = next((
            cell for cell in scenario.grid.searchable_cells
            if cell.cell_id == distractor_cell_id
        ), None)
        distractor_visible = (
            distractor_cell is not None
            and distractor_cell.cell_id in visible_ids
        )
        if (
            distractor_visible
            and _stable_unit_interval(
                seed,
                scenario.scenario_id,
                viewpoint.key,
                "persistent-distractor",
            ) < self.config.persistent_distractor_probability
        ):
            detections.append(self._localized_detection(
                scenario,
                viewpoint,
                distractor_cell.center,
                f"persistent-distractor:{scenario.scenario_id}",
                "persistent_distractor",
                False,
                seed,
            ))

        if not target_visible:
            background_kind = self._background_false_alarm_kind(
                scenario,
                viewpoint,
                seed,
            )
            if background_kind is not None:
                shared_identity = (
                    background_kind == "correlated_false_alarm"
                    and self.config.correlated_false_alarm_shared_identity
                )
                entity_id = (
                    f"correlated-false-alarm:{scenario.scenario_id}"
                    if shared_identity
                    else f"false-positive:{scenario.scenario_id}:{viewpoint.key}"
                )
                detections.append(self._localized_detection(
                    scenario,
                    viewpoint,
                    (viewpoint.x, viewpoint.y),
                    entity_id,
                    background_kind,
                    False,
                    seed,
                ))
        return tuple(detections)

    def _background_false_alarm_kind(
        self,
        scenario: SearchBenchmarkScenario,
        viewpoint: Viewpoint,
        seed: int,
    ) -> Optional[str]:
        common_mode = _stable_unit_interval(
            seed,
            scenario.scenario_id,
            "false-alarm-common-mode",
        ) < self.config.false_alarm_correlation
        if common_mode:
            detected = _stable_unit_interval(
                seed,
                scenario.scenario_id,
                "false-alarm-common-event",
            ) < self.config.sensor_model.false_positive_probability
            return "correlated_false_alarm" if detected else None
        detected = _stable_unit_interval(
            seed,
            scenario.scenario_id,
            viewpoint.key,
            "sensor",
        ) < self.config.sensor_model.false_positive_probability
        return "independent_false_alarm" if detected else None

    def _localized_detection(
        self,
        scenario: SearchBenchmarkScenario,
        viewpoint: Viewpoint,
        source_xy: Tuple[float, float],
        entity_id: str,
        source_kind: str,
        ground_truth_match: bool,
        seed: int,
    ) -> TargetDetection:
        error_x, error_y = _stable_normal_pair(
            self.config.localization_error_std_m,
            seed,
            scenario.scenario_id,
            viewpoint.key,
            source_kind,
        )
        estimated_x = source_xy[0] + error_x
        estimated_y = source_xy[1] + error_y
        localized_cell = scenario.grid.cell_at(estimated_x, estimated_y)
        localization_error = math.hypot(error_x, error_y)
        return TargetDetection(
            label=scenario.task.target.query,
            confidence=self.config.detection_confidence,
            estimated_position=(estimated_x, estimated_y, 0.0),
            entity_id=entity_id,
            attributes={
                "source_kind": source_kind,
                "ground_truth_match": ground_truth_match,
                "localized_cell_id": (
                    localized_cell.cell_id
                    if localized_cell is not None and localized_cell.searchable
                    else None
                ),
                "localization_error_m": localization_error,
                "localization_error_xy_m": (error_x, error_y),
            },
        )

    def _shortest_detection_distance(
        self,
        scenario: SearchBenchmarkScenario,
        candidates: Tuple[ViewpointCandidate, ...],
    ) -> float:
        start = self.config.start_viewpoint(scenario)
        distances = [
            _viewpoint_distance(start, candidate.viewpoint)
            for candidate in candidates
            if scenario.target_cell_id in candidate.visible_cell_ids
        ]
        return min(distances, default=0.0)


class SearchBenchmarkRunner:
    """Evaluate every configured policy on every scenario and repetition."""

    def __init__(self, config: SearchBenchmarkConfig) -> None:
        self.config = config
        self.episode_runner = SearchEpisodeRunner(config)

    def run(
        self,
        scenarios: Iterable[SearchBenchmarkScenario],
    ) -> SearchBenchmarkReport:
        scenario_items = tuple(scenarios)
        if not scenario_items:
            raise ValueError("at least one benchmark scenario is required")
        if len({item.scenario_id for item in scenario_items}) != len(scenario_items):
            raise ValueError("benchmark scenario IDs must be unique")
        episodes = tuple(
            self.episode_runner.run(scenario, policy_name, repetition)
            for scenario in scenario_items
            for repetition in range(self.config.repetitions)
            for policy_name in self.config.policy_names
        )
        return SearchBenchmarkReport(
            config=self.config,
            scenario_ids=tuple(item.scenario_id for item in scenario_items),
            episodes=episodes,
            aggregates=aggregate_policy_results(
                episodes,
                self.config.policy_names,
            ),
            condition_aggregates=tuple(
                aggregate
                for condition in dict.fromkeys(
                    item.prior_condition for item in scenario_items
                )
                for aggregate in aggregate_policy_results(
                    episodes,
                    self.config.policy_names,
                    prior_condition=condition,
                )
            ),
        )


def _viewpoint_distance(first: Viewpoint, second: Viewpoint) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def _spl(success: bool, shortest_distance: float, actual_distance: float) -> float:
    if not success:
        return 0.0
    if shortest_distance <= 0:
        return 1.0 if actual_distance <= 1e-9 else 0.0
    return shortest_distance / max(shortest_distance, actual_distance)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        ":".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _stable_unit_interval(*parts: object) -> float:
    return _stable_seed(*parts) / float(2 ** 64)


def _stable_normal_pair(
    standard_deviation: float,
    *parts: object,
) -> Tuple[float, float]:
    if standard_deviation <= 0:
        return (0.0, 0.0)
    first = max(_stable_unit_interval(*parts, "normal-radius"), 1e-12)
    second = _stable_unit_interval(*parts, "normal-angle")
    radius = standard_deviation * math.sqrt(-2.0 * math.log(first))
    angle = 2.0 * math.pi * second
    return (radius * math.cos(angle), radius * math.sin(angle))


def _detection_trace(detection: TargetDetection) -> Mapping[str, Any]:
    return {
        "label": detection.label,
        "confidence": detection.confidence,
        "entity_id": detection.entity_id,
        "estimated_position": detection.estimated_position,
        "attributes": dict(detection.attributes),
    }
