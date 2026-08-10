import json
from gsi_search_bridge.building_route_planner import (
    BuildingObstacle,
    load_building_obstacles,
    plan_building_avoiding_route,
    point_has_building_clearance,
    segment_intersects_obstacle,
)


def _building(
    obstacle_id="building",
    min_x=4.0,
    min_y=-2.0,
    max_x=6.0,
    max_y=2.0,
    min_z=0.0,
    max_z=50.0,
):
    return BuildingObstacle(
        obstacle_id, min_x, min_y, max_x, max_y, min_z, max_z
    )


def test_segment_intersection_includes_boundary_contact():
    obstacle = _building()

    assert segment_intersects_obstacle((0.0, 0.0, 30.0), (10.0, 0.0, 30.0), obstacle)
    assert segment_intersects_obstacle((0.0, 2.0, 30.0), (10.0, 2.0, 30.0), obstacle)
    assert not segment_intersects_obstacle((0.0, 3.0, 30.0), (10.0, 3.0, 30.0), obstacle)


def test_direct_route_is_preserved_when_clear():
    goal = (10.0, 5.0, 30.0)

    assert plan_building_avoiding_route(
        (0.0, 5.0, 30.0),
        goal,
        (_building(),),
        horizontal_clearance_m=1.0,
        vertical_clearance_m=2.0,
    ) == (goal,)


def test_route_uses_safe_corners_when_building_blocks_segment():
    obstacle = _building()
    route = plan_building_avoiding_route(
        (0.0, 0.0, 30.0),
        (10.0, 0.0, 30.0),
        (obstacle,),
        horizontal_clearance_m=1.0,
        vertical_clearance_m=2.0,
        corner_offset_m=0.1,
    )

    assert route is not None
    assert len(route) == 3
    assert route[-1] == (10.0, 0.0, 30.0)
    inflated = obstacle.inflated(1.0)
    start = (0.0, 0.0, 30.0)
    for waypoint in route:
        assert not segment_intersects_obstacle(start, waypoint, inflated)
        start = waypoint


def test_low_building_is_ignored_above_vertical_clearance():
    goal = (10.0, 0.0, 30.0)
    low_building = _building(max_z=20.0)

    assert plan_building_avoiding_route(
        (0.0, 0.0, 30.0),
        goal,
        (low_building,),
        horizontal_clearance_m=1.0,
        vertical_clearance_m=2.0,
    ) == (goal,)


def test_viewpoint_inside_clearance_is_rejected():
    obstacle = _building()

    assert not point_has_building_clearance(
        (3.5, 0.0, 30.0),
        (obstacle,),
        horizontal_clearance_m=1.0,
        vertical_clearance_m=2.0,
    )
    assert point_has_building_clearance(
        (3.5, 0.0, 60.0),
        (obstacle,),
        horizontal_clearance_m=1.0,
        vertical_clearance_m=2.0,
    )


def test_route_does_not_leave_configured_bounds():
    obstacle = _building(min_y=0.0, max_y=2.0)

    assert plan_building_avoiding_route(
        (0.0, 1.0, 30.0),
        (10.0, 1.0, 30.0),
        (obstacle,),
        horizontal_clearance_m=1.0,
        vertical_clearance_m=2.0,
        corner_offset_m=0.1,
        route_bounds=(0.0, 0.0, 10.0, 3.0),
    ) is None


def test_semantic_map_loader_reads_only_rectangular_buildings(tmp_path):
    semantic_map = tmp_path / "map.json"
    semantic_map.write_text(json.dumps({"nodes": [
        {
            "id": "b1",
            "properties": {
                "category": "building",
                "elevation_min_m": 1.0,
                "elevation_max_m": 12.0,
            },
            "shape": {
                "type": "rectangle",
                "min_corner": [5.0, 6.0],
                "max_corner": [1.0, 2.0],
            },
        },
        {
            "id": "road",
            "properties": {"category": "transportation_facility"},
            "shape": {
                "type": "rectangle",
                "min_corner": [0.0, 0.0],
                "max_corner": [9.0, 9.0],
            },
        },
    ]}), encoding="utf-8")

    assert load_building_obstacles(str(semantic_map)) == (
        BuildingObstacle("b1", 1.0, 2.0, 5.0, 6.0, 1.0, 12.0),
    )
