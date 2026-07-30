import json
import math
import unittest

from modules.search_intelligence import (
    CandidateViewpointGenerator,
    CoveragePolicy,
    SearchObservation,
    SearchArea,
    SearchGrid,
    SearchSession,
    SearchTask,
    viewpoint_distance_matrix,
)


class SearchGridTests(unittest.TestCase):
    def setUp(self):
        self.task = SearchTask.from_skill_params({
            "task_id": "grid-1",
            "area_token": "Parking-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [80, 0], [80, 60], [0, 60]],
            },
            "target_token": "yellow-van",
        })

    def test_rectangle_is_discretized_into_stable_row_major_cells(self):
        grid = SearchGrid.from_task(self.task, resolution_m=20.0)

        self.assertEqual((grid.width, grid.height), (4, 3))
        self.assertEqual(len(grid.cells), 12)
        self.assertEqual(len(grid.searchable_cells), 12)
        self.assertEqual(grid.cell(1, 2).cell_id, "Parking-A:r1:c2")
        self.assertEqual(grid.cell(1, 2).center, (50.0, 30.0))
        self.assertEqual(grid.cell_at(80.0, 60.0).cell_id, "Parking-A:r2:c3")

    def test_polygon_outside_cells_and_explicit_exclusions_are_unsearchable(self):
        area = SearchArea(
            area_id="L-shaped",
            geometry={
                "kind": "area",
                "coords": [[0, 0], [40, 0], [40, 20], [20, 20], [20, 40], [0, 40]],
            },
        )
        excluded = ({
            "kind": "rectangle",
            "coords": [[0, 0], [20, 0], [20, 20], [0, 20]],
        },)

        grid = SearchGrid.from_area(area, 20.0, excluded_geometries=excluded)

        self.assertEqual(len(grid.cells), 4)
        self.assertFalse(grid.cell(0, 0).searchable)
        self.assertTrue(grid.cell(1, 0).searchable)
        self.assertFalse(grid.cell(1, 1).searchable)

    def test_uniform_belief_only_contains_searchable_cells(self):
        grid = SearchGrid.from_task(self.task, resolution_m=20.0)

        belief = grid.uniform_belief()

        self.assertEqual(set(belief), {cell.cell_id for cell in grid.searchable_cells})
        self.assertAlmostEqual(sum(belief.values()), 1.0)
        json.dumps(grid.to_dict())

    def test_circle_grid_uses_cell_center_inclusion(self):
        area = SearchArea(
            area_id="circle",
            geometry={"kind": "circle", "center": [0, 0], "radius": 20},
        )

        grid = SearchGrid.from_area(area, resolution_m=10.0)

        self.assertEqual((grid.width, grid.height), (4, 4))
        self.assertEqual(len(grid.searchable_cells), 12)

    def test_buffered_point_and_axis_aligned_line_have_valid_grids(self):
        point_grid = SearchGrid.from_area(
            SearchArea(
                area_id="point",
                geometry={"kind": "point", "coords": [[10, 10]], "buffer": 10},
            ),
            resolution_m=10.0,
        )
        line_grid = SearchGrid.from_area(
            SearchArea(
                area_id="line",
                geometry={
                    "kind": "line",
                    "coords": [[0, 0], [40, 0]],
                    "buffer": 10,
                },
            ),
            resolution_m=10.0,
        )

        self.assertEqual((point_grid.width, point_grid.height), (2, 2))
        self.assertTrue(point_grid.searchable_cells)
        self.assertEqual((line_grid.width, line_grid.height), (6, 2))
        self.assertTrue(line_grid.searchable_cells)

    def test_invalid_or_empty_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            SearchGrid.from_area(
                SearchArea("unknown", {"kind": "unsupported"}),
                resolution_m=10.0,
            )


class CandidateViewpointTests(unittest.TestCase):
    def setUp(self):
        task = SearchTask.from_skill_params({
            "task_id": "candidate-1",
            "area_token": "Test-Area",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [60, 0], [60, 60], [0, 60]],
            },
            "target_token": "target",
        })
        self.grid = SearchGrid.from_task(task, resolution_m=20.0)

    def test_candidates_include_pose_anchor_and_visible_cells(self):
        generator = CandidateViewpointGenerator(
            altitude_m=30.0,
            footprint_radius_m=21.0,
        )

        candidates = generator.generate(self.grid)
        center = next(item for item in candidates if item.anchor_cell_id == "Test-Area:r1:c1")

        self.assertEqual(len(candidates), 9)
        self.assertEqual(center.viewpoint.z, 30.0)
        self.assertEqual(center.viewpoint.pitch, -math.pi / 2.0)
        self.assertEqual(len(center.visible_cell_ids), 5)
        self.assertIn(center.anchor_cell_id, center.visible_cell_ids)
        json.dumps(center.to_dict())

    def test_stride_and_limit_bound_candidate_count(self):
        candidates = CandidateViewpointGenerator(
            altitude_m=20.0,
            stride_cells=2,
            max_candidates=3,
        ).generate(self.grid)

        self.assertEqual(len(candidates), 3)

    def test_distance_matrix_is_symmetric_with_zero_diagonal(self):
        candidates = CandidateViewpointGenerator(
            altitude_m=20.0,
            footprint_radius_m=10.0,
            max_candidates=3,
        ).generate(self.grid)

        distances = viewpoint_distance_matrix(candidates)

        self.assertEqual(len(distances), 3)
        self.assertEqual(distances[0][0], 0.0)
        self.assertEqual(distances[0][1], distances[1][0])
        self.assertAlmostEqual(distances[0][1], 20.0)

    def test_grid_coverage_advances_the_common_search_session(self):
        task = SearchTask.from_skill_params({
            "task_id": "grid-session",
            "area_token": "Test-Area",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [60, 0], [60, 60], [0, 60]],
            },
            "target_token": "target",
        })
        session = SearchSession(
            task,
            CoveragePolicy(pass_spacing_m=20.0, altitude_m=30.0),
            initial_belief=self.grid.uniform_belief(),
        )

        viewpoint = session.next_viewpoint()
        visible_cells = self.grid.cells_within_radius(
            viewpoint.x,
            viewpoint.y,
            radius_m=30.0,
        )
        state = session.record_observation(SearchObservation(
            viewpoint=viewpoint,
            timestamp_s=1.0,
            visible_cell_ids=tuple(cell.cell_id for cell in visible_cells),
        ))

        self.assertEqual(set(state.belief), set(self.grid.uniform_belief()))
        self.assertEqual(
            set(state.observed_cell_quality),
            {cell.cell_id for cell in visible_cells},
        )


if __name__ == "__main__":
    unittest.main()
