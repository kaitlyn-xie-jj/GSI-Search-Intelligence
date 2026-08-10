import math
import unittest
from dataclasses import replace

from modules.search_intelligence import (
    CoveragePolicy,
    SearchObservation,
    SearchGrid,
    SearchState,
    SearchTask,
    Viewpoint,
)


class CoveragePolicyTests(unittest.TestCase):
    def setUp(self):
        self.task = SearchTask.from_skill_params({
            "task_id": "coverage-1",
            "area_token": "Parking-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [80, 0], [80, 60], [0, 60]],
            },
            "target_token": "yellow-van",
            "target": {
                "class": "vehicle",
                "type": "van",
                "features": {"color": "yellow"},
            },
        })
        self.state = SearchState.initial(self.task, belief={})

    def test_generates_contract_viewpoints_from_existing_zigzag_planner(self):
        policy = CoveragePolicy(pass_spacing_m=20.0, altitude_m=25.0)

        viewpoints = policy.plan(self.state)

        self.assertGreaterEqual(len(viewpoints), 4)
        self.assertTrue(all(isinstance(item, Viewpoint) for item in viewpoints))
        self.assertTrue(all(item.z == 25.0 for item in viewpoints))
        self.assertTrue(all(item.pitch == -math.pi / 2.0 for item in viewpoints))
        self.assertNotEqual(viewpoints[0].yaw, viewpoints[1].yaw)

    def test_select_next_skips_viewpoint_recorded_in_search_state(self):
        policy = CoveragePolicy(pass_spacing_m=20.0, altitude_m=25.0)
        first = policy.select_next(self.state)
        self.assertIsNotNone(first)

        observed = self.state.advance(
            SearchObservation(viewpoint=first, timestamp_s=1.0)
        )

        second = policy.select_next(observed)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.key, second.key)
        self.assertEqual(len(policy.plan(observed)), len(policy.plan(self.state)) - 1)

    def test_uses_current_uav_altitude_when_not_configured(self):
        current = Viewpoint(0.0, 0.0, 42.0, 0.0)
        state = SearchState.initial(self.task, belief={}, current_viewpoint=current)

        viewpoints = CoveragePolicy(
            pass_spacing_m=20.0,
            route_start_hint=current,
        ).plan(state)

        self.assertTrue(viewpoints)
        self.assertTrue(all(item.z == 42.0 for item in viewpoints))

    def test_starts_from_nearest_route_endpoint(self):
        current = Viewpoint(100.0, 100.0, 25.0, 0.0)
        state = SearchState.initial(
            self.task, belief={}, current_viewpoint=current
        )

        viewpoints = CoveragePolicy(
            pass_spacing_m=20.0,
            route_start_hint=current,
        ).plan(state)

        self.assertLess(
            math.hypot(viewpoints[0].x - current.x, viewpoints[0].y - current.y),
            math.hypot(viewpoints[-1].x - current.x, viewpoints[-1].y - current.y),
        )

        advanced = state.advance(
            SearchObservation(viewpoint=viewpoints[0], timestamp_s=1.0)
        )
        remaining = CoveragePolicy(
            pass_spacing_m=20.0,
            route_start_hint=current,
        ).plan(advanced)
        self.assertEqual(remaining[0].key, viewpoints[1].key)

    def test_filters_unsafe_viewpoints(self):
        viewpoints = CoveragePolicy(
            pass_spacing_m=20.0,
            viewpoint_filter=lambda item: item.x < 50.0,
        ).plan(self.state)

        self.assertTrue(viewpoints)
        self.assertTrue(all(item.x < 50.0 for item in viewpoints))

    def test_recovery_visits_uncovered_cells_after_primary_route(self):
        grid = SearchGrid.from_task(self.task, resolution_m=20.0)
        policy = CoveragePolicy(
            pass_spacing_m=20.0,
            altitude_m=25.0,
            search_grid=grid,
            recovery_enabled=True,
            recovery_offset_m=5.0,
        )
        primary = policy.plan(self.state)
        state = replace(
            self.state,
            visited_viewpoint_keys=tuple(item.key for item in primary),
            observed_cell_quality={grid.searchable_cells[0].cell_id: 1.0},
        )

        recovery = policy.plan(state)
        metadata = policy.decision_metadata(state, recovery[0])

        self.assertTrue(recovery)
        self.assertEqual(metadata["coverage_phase"], "recovery")
        self.assertEqual(metadata["coverage_covered_cells"], 1)
        self.assertEqual(
            metadata["coverage_deferred_cells"],
            len(grid.searchable_cells) - 1,
        )

    def test_recovery_finishes_when_all_searchable_cells_are_covered(self):
        grid = SearchGrid.from_task(self.task, resolution_m=20.0)
        policy = CoveragePolicy(
            pass_spacing_m=20.0,
            search_grid=grid,
            recovery_enabled=True,
        )
        primary = policy.plan(self.state)
        state = replace(
            self.state,
            visited_viewpoint_keys=tuple(item.key for item in primary),
            observed_cell_quality={
                cell.cell_id: 1.0 for cell in grid.searchable_cells
            },
        )

        self.assertEqual(policy.plan(state), ())

    def test_unsupported_geometry_finishes_without_action(self):
        task = SearchTask.from_skill_params({
            "task_id": "invalid-area",
            "area_token": "Unknown",
            "area": {"kind": "unsupported"},
            "target_token": "target",
        })
        state = SearchState.initial(task, belief={})

        self.assertIsNone(CoveragePolicy().select_next(state))

    def test_policy_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            CoveragePolicy(pass_spacing_m=0.0)
        with self.assertRaises(ValueError):
            CoveragePolicy(observation_spacing_m=0.0)

    def test_optional_observation_sampling_bounds_segment_length(self):
        viewpoints = CoveragePolicy(
            pass_spacing_m=20.0,
            altitude_m=25.0,
            observation_spacing_m=15.0,
        ).plan(self.state)

        distances = [
            math.hypot(second.x - first.x, second.y - first.y)
            for first, second in zip(viewpoints, viewpoints[1:])
        ]
        self.assertTrue(distances)
        self.assertLessEqual(max(distances), 15.0 + 1e-9)

    def test_select_next_stops_when_search_budget_is_exhausted(self):
        task = SearchTask.from_skill_params({
            "task_id": "budgeted-coverage",
            "area_token": "Parking-A",
            "area": self.task.search_area.geometry,
            "target_token": "yellow-van",
            "max_viewpoints": 1,
        })
        state = SearchState.initial(task, belief={})
        first = CoveragePolicy(pass_spacing_m=20.0).select_next(state)
        self.assertIsNotNone(first)
        state = state.advance(SearchObservation(viewpoint=first, timestamp_s=1.0))

        self.assertEqual(state.exhausted_budget, "viewpoints")
        self.assertIsNone(CoveragePolicy(pass_spacing_m=20.0).select_next(state))

    def test_plan_is_capped_by_remaining_viewpoint_budget(self):
        task = SearchTask.from_skill_params({
            "task_id": "plan-budget",
            "area_token": "Parking-A",
            "area": self.task.search_area.geometry,
            "target_token": "yellow-van",
            "max_viewpoints": 2,
        })
        state = SearchState.initial(task, {})

        viewpoints = CoveragePolicy(pass_spacing_m=20.0).plan(state)

        self.assertEqual(len(viewpoints), 2)


if __name__ == "__main__":
    unittest.main()
