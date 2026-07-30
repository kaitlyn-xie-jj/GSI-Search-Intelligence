"""Closed-loop episode and suite runners for policy comparison."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, Mapping, Tuple

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
            policy_trace=session.policy_decisions,
            belief_entropy_trace=tuple(entropy_trace),
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
            )
        raise ValueError(f"unsupported benchmark policy: {policy_name}")

    def _detections(
        self,
        scenario: SearchBenchmarkScenario,
        viewpoint: Viewpoint,
        visible_cell_ids: Tuple[str, ...],
        seed: int,
    ) -> Tuple[TargetDetection, ...]:
        target_visible = scenario.target_cell_id in set(visible_cell_ids)
        probability = (
            self.config.sensor_model.effective_detection_probability(
                self.config.observation_quality
            )
            if target_visible
            else self.config.sensor_model.false_positive_probability
        )
        sample = _stable_unit_interval(
            seed,
            scenario.scenario_id,
            viewpoint.key,
            "sensor",
        )
        if sample >= probability:
            return ()
        if target_visible:
            target = scenario.target_cell.center
            return (TargetDetection(
                label=scenario.task.target.query,
                confidence=self.config.detection_confidence,
                estimated_position=(target[0], target[1], 0.0),
                entity_id=scenario.target_entity_id,
                attributes={
                    "ground_truth_match": True,
                    "localized_cell_id": scenario.target_cell_id,
                },
            ),)
        localized_cell = scenario.grid.nearest_searchable_cell(
            viewpoint.x,
            viewpoint.y,
        )
        return (TargetDetection(
            label=scenario.task.target.query,
            confidence=self.config.detection_confidence,
            estimated_position=(viewpoint.x, viewpoint.y, 0.0),
            entity_id=(
                f"false-positive:{scenario.scenario_id}:{viewpoint.key}"
            ),
            attributes={
                "ground_truth_match": False,
                "localized_cell_id": (
                    localized_cell.cell_id if localized_cell is not None else None
                ),
            },
        ),)

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
