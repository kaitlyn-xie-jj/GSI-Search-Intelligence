import asyncio
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import logging
from enum import Enum
from modules.config.base.enums import SkillName
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.semantic_platform.scene_graph_manager import SemanticSceneGraph
from modules.platform.semantic_platform.skill_simulator_enhance import EnhancedSkillMotionSimulator
from modules.platform.semantic_platform.visualization import RealTimeScenarioVisualizer
from modules.search_intelligence import (
    ActiveSearchPolicy,
    BayesianBeliefUpdater,
    BinarySensorModel,
    CandidateViewpointGenerator,
    CoveragePolicy,
    GreedyPriorPolicy,
    RandomPolicy,
    SearchObservation,
    SearchGrid,
    SearchPrior,
    SearchSession,
    SearchTask,
    SemanticGridBuilder,
    TargetDetection,
    Viewpoint,
)
from modules.utils.location_utils import (
    shape_center_point
)

logger = logging.getLogger(__name__)

class MotionController:
    """
    Motion controller - manages motion simulation and visualization updates for all entities.
    """

    def __init__(self, motion_simulator: EnhancedSkillMotionSimulator, visualizer: RealTimeScenarioVisualizer, scene_graph: AbstractSceneGraph, enable_visualization: bool = False):
        """
        Initialize the motion controller.

        Args:
            motion_simulator: EnhancedSkillMotionSimulator instance
            visualizer: RealTimeScenarioVisualizer instance
            scene_graph: scene graph for entity lookups
        """
        self.motion_simulator = motion_simulator
        self.visualizer = visualizer
        self.scene_graph = scene_graph
        self.enable_visualization = bool(enable_visualization)

        # Entity motion state tracking
        self._entity_motion_states = {}  # entity_id -> current_position

        # Autonomous motion tracking
        self._autonomous_motions_in_progress = {}  # entity_id -> motion_task

        # Lookup maps (from scene_graph)
        self.label_to_id_map = scene_graph.get_node_map(map_type='label_to_id')
        self.id_to_label_map = scene_graph.get_node_map(map_type='id_to_label')

    def _can_viz(self) -> bool:
        return bool(self.enable_visualization and self.visualizer)

    async def simulate_skill_motion(
        self,
        robot_id: int,
        skill: str,
        params: Dict,
        interrupt_check: Optional[Callable] = None
    ) -> Dict:
        """
        Simulate skill execution motion - reads target/area/etc. uniformly from params.
        """
        # Get robot info
        robot = self.scene_graph.get_robot(robot_id)
        if not robot:
            logger.error(f"Cannot find robot with ID: {robot_id}")
            return {'success': False, 'outcome': 'robot_not_found'}

        robot_type = robot.get('properties', {}).get('type')
        robot_label = robot.get('properties', {}).get('label')

        # Validate skill availability for this robot type
        skill_validation = {
            'UAV': ['take_off', 'return_home', 'navigate', 'take_photo', 'follow', 'broadcast', 'handle_hazard', 'search'],
            'FW_UAV': ['take_off', 'return_home', 'navigate', 'search'],
            'UGV': ['navigate', 'broadcast'], 
            'Quadruped': ['navigate', 'follow', 'take_photo', 'search', 'guide', 'broadcast'], 
            'Humanoid': ['navigate', 'place'] 
        }
        skill_name = skill.split('.')[1] if '.' in skill else skill
        if robot_type in skill_validation and skill_name not in skill_validation[robot_type]:
            logger.error(f"Skill {skill_name} not available for robot type {robot_type}")
            return {'success': False, 'outcome': 'skill_not_available'}

        # Get robot current position
        robot_pos = self.get_entity_position(robot_id)
        if not robot_pos:
            logger.error(f"Cannot get position for robot {robot_label}")
            return {'success': False, 'outcome': 'position_unknown'}

        # Default interrupt check
        if interrupt_check is None:
            interrupt_check = lambda: False

        # Create update callback
        def update_callback(motion_data: Dict):
            self._update_skill_visualization(
                robot_id, robot_label, skill, motion_data
            )

        # Dispatch by skill type
        try:
            if skill == SkillName.NAVIGATE.value:
                result = await self._simulate_navigate(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.TAKE_PHOTO.value:
                result = await self._simulate_take_photo(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.TAKE_OFF.value:
                result = await self._simulate_take_off(
                    robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.RETURN_HOME.value:
                result = await self._simulate_return_home(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.FOLLOW.value:
                result = await self._simulate_follow(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.SEARCH.value:
                result = await self._simulate_search(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.BROADCAST.value:
                result = await self._simulate_broadcast(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.PLACE.value:
                if robot_type != 'Humanoid':
                    return {'success': False, 'outcome': 'wrong_robot_type'}
                result = await self._simulate_place(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.HANDLE_HAZARD.value:
                result = await self._simulate_handle_hazard(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            elif skill == SkillName.GUIDE.value:
                result = await self._simulate_guide(
                    robot_id, robot_type, robot_pos,
                    params, update_callback, interrupt_check
                )
            else:
                # Unknown skill
                result = {'success': False, 'outcome': 'unknown_skill'}

            return result

        except Exception as e:
            # logger.error(f"Error during skill simulation: {e}")
            logger.exception(f"Error during skill simulation (robot_id={robot_id}, skill={skill}): {e}")
            return {'success': False, 'outcome': 'simulation_error', 'error': str(e)}

    async def _simulate_navigate(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulate navigate skill (destination resolved from params)."""
        dest_xy = None

        # 1) Explicit coordinates
        d = params.get('dest')
        if isinstance(d, dict) and isinstance(d.get('x'), (int, float)) and isinstance(d.get('y'), (int, float)):
            dest_xy = [float(d['x']), float(d['y'])]

        # 2) Derive from area
        if dest_xy is None and isinstance(params.get('area'), dict):
            a = params['area']; k = a.get('kind')
            if k == 'point' and a.get('coords'):
                dest_xy = [float(a['coords'][0][0]), float(a['coords'][0][1])]
            elif k == 'circle' and a.get('center'):
                dest_xy = [float(a['center'][0]), float(a['center'][1])]
            elif k == 'area' and a.get('coords'):
                xs = [c[0] for c in a['coords']]; ys = [c[1] for c in a['coords']]
                dest_xy = [sum(xs)/len(xs), sum(ys)/len(ys)]

        # 3) Fall back to object_id
        tgt_id = params.get('object_id')
        if dest_xy is None and tgt_id is not None:
            pos = self.get_entity_position(tgt_id)
            if pos:
                dest_xy = pos

        if dest_xy is None:
            return {'success': False, 'outcome': 'no_destination_specified'}

        obstacles = self._get_obstacles_info(robot_id, tgt_id)
        exec_time = params.get('execution_time', 3.0)

        # Update carried object positions during navigation
        def navigate_update_callback(motion_data: Dict):
            update_callback(motion_data)

            # Sync objects connected by the "carrying" relation
            if 'position' in motion_data:
                carried_ids = self.scene_graph.get_neighbors_by_relation(robot_id, 'carrying')
                for carried_id in carried_ids:
                    self._entity_motion_states[carried_id] = motion_data['position']
                    prop = self.scene_graph.get_prop(carried_id)
                    prop_label = prop.get('properties', {}).get('label', '') if prop else ''
                    if self._can_viz():
                        self.visualizer.update_entity_position(
                            entity_id=carried_id,
                            position=motion_data['position'],
                            additional_info={'label': prop_label, 'carried': True}
                        )

        result = await self.motion_simulator.simulate_navigate(
            robot_type=robot_type,
            start_pos=robot_pos,
            end_pos=dest_xy,
            obstacles=obstacles,
            update_callback=navigate_update_callback,
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )

        if result['success']:
            self._entity_motion_states[robot_id] = result['final_position']

        return result

    async def _simulate_take_photo(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulate the take-photo skill (target is derived from params.object_id or area)."""

        target_pos = None
        oid = params.get('object_id')
        if oid is not None:
            target_pos = self.get_entity_position(oid)

        if target_pos is None and isinstance(params.get('area'), dict):
            a = params['area']; k = a.get('kind')
            if k == 'point' and a.get('coords'):
                target_pos = [float(a['coords'][0][0]), float(a['coords'][0][1])]
            elif k == 'circle' and a.get('center'):
                target_pos = [float(a['center'][0]), float(a['center'][1])]
            elif k == 'area' and a.get('coords'):
                xs = [c[0] for c in a['coords']]; ys = [c[1] for c in a['coords']]
                target_pos = [sum(xs)/len(xs), sum(ys)/len(ys)]

        if target_pos is None:
            target_pos = robot_pos.copy() # If no target exists, default to the current position

        exec_time = params.get('execution_time', 1.5)

        # Photo type by robot type
        if robot_type in ['UAV', 'FW_UAV']:
            photo_type = 'aerial_wide'
        elif robot_type == 'Quadruped':
            photo_type = 'ground_detail'
        else:
            photo_type = 'generic'

        result = await self.motion_simulator.simulate_take_photo(
            robot_type=robot_type,
            robot_pos=robot_pos,
            target_pos=target_pos,
            update_callback=lambda data: update_callback({**data, 'photo_type': photo_type}),
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )

        result['photo_type'] = photo_type
        return result

    async def _simulate_search(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """
        Simulate the search or patrol skill (area comes from params.area).
        """
        area_geom = params.get('area')
        if not isinstance(area_geom, dict):
            logger.error("Search/Patrol skill requires 'area' geometry dict")
            return {'success': False, 'outcome': 'missing_area'}

        search_type = params.get('goal_type', 'area_search')
        exec_time = params.get('execution_time', 8.0)
        pass_spacing = params.get('pass_spacing', 40.0)

        # 1. Generate the baseline route through the common search-policy contract.
        search_task = SearchTask.from_skill_params(params)
        flight_altitude = float(robot_pos[2]) if len(robot_pos) > 2 else 30.0
        coverage_policy = CoveragePolicy(
            pass_spacing_m=pass_spacing,
            altitude_m=flight_altitude,
            camera_pitch_rad=float(params.get('camera_pitch_rad', -np.pi / 2.0)),
            max_points=int(params.get('max_search_points', 3000)),
        )
        search_grid = SearchGrid.from_task(
            search_task,
            resolution_m=float(params.get('grid_resolution_m', pass_spacing)),
            excluded_geometries=params.get('excluded_geometries') or (),
        )
        semantic_builder = SemanticGridBuilder(
            line_buffer_m=float(params.get(
                'semantic_line_buffer_m', search_grid.resolution_m / 2.0
            )),
            point_buffer_m=float(params.get(
                'semantic_point_buffer_m', search_grid.resolution_m / 2.0
            )),
        )
        search_grid = semantic_builder.from_scene_graph(search_grid, self.scene_graph)

        raw_search_prior = params.get('search_prior')
        prior_projection = None
        if isinstance(raw_search_prior, dict):
            prior_projection = SearchPrior.from_llm_output(
                search_task.task_id,
                raw_search_prior,
            ).project(search_grid)
            initial_belief = prior_projection.belief
            initial_policy_metadata = {
                'prior_source': 'llm_semantic',
                'prior_confidence': prior_projection.confidence,
                'matched_prior_labels': prior_projection.matched_labels,
                'unmatched_prior_labels': prior_projection.unmatched_labels,
            }
        else:
            initial_belief = search_grid.uniform_belief()
            initial_policy_metadata = {'prior_source': 'uniform'}
        footprint_radius = params.get('sensor_footprint_radius_m')
        max_candidates = params.get('max_search_candidates')
        camera_model = CandidateViewpointGenerator(
            altitude_m=flight_altitude,
            horizontal_fov_rad=float(params.get('camera_fov_rad', np.pi / 2.0)),
            footprint_radius_m=(
                float(footprint_radius) if footprint_radius is not None else None
            ),
            stride_cells=int(params.get('candidate_stride_cells', 1)),
            max_candidates=(
                int(max_candidates) if max_candidates is not None else None
            ),
        )
        sensor_model = BinarySensorModel(
            detection_probability=float(params.get(
                'sensor_detection_probability', 0.85
            )),
            false_positive_probability=float(params.get(
                'sensor_false_positive_probability', 0.05
            )),
        )
        policy_name = str(params.get('search_policy', 'coverage')).strip().lower()
        if search_type == 'patrol':
            policy_name = 'coverage'

        if policy_name == 'coverage':
            search_policy = coverage_policy
        else:
            candidates = camera_model.generate(search_grid)
            if not candidates:
                return {'success': False, 'outcome': 'candidate_generation_failed'}
            if policy_name == 'active':
                minimum_utility = params.get('minimum_viewpoint_utility')
                search_policy = ActiveSearchPolicy(
                    candidates,
                    sensor_model=sensor_model,
                    detection_weight=float(params.get('detection_weight', 1.0)),
                    information_gain_weight=float(params.get(
                        'information_gain_weight', 1.0
                    )),
                    novelty_weight=float(params.get('novelty_weight', 0.25)),
                    travel_weight=float(params.get('travel_weight', 0.1)),
                    distance_scale_m=float(params.get('distance_scale_m', 100.0)),
                    minimum_utility=(
                        float(minimum_utility)
                        if minimum_utility is not None
                        else None
                    ),
                )
            elif policy_name == 'greedy_prior':
                search_policy = GreedyPriorPolicy(
                    candidates,
                    initial_belief,
                    distance_weight=float(params.get(
                        'prior_distance_weight', 0.0
                    )),
                    distance_scale_m=float(params.get('distance_scale_m', 100.0)),
                )
            elif policy_name == 'random':
                search_policy = RandomPolicy(
                    candidates,
                    seed=int(params.get('search_seed', 0)),
                )
            else:
                return {
                    'success': False,
                    'outcome': 'unsupported_search_policy',
                    'search_policy': policy_name,
                }

        initial_policy_metadata['search_policy'] = policy_name
        search_session = SearchSession(
            search_task,
            search_policy,
            initial_belief=initial_belief,
            current_viewpoint=Viewpoint(
                x=float(robot_pos[0]),
                y=float(robot_pos[1]),
                z=flight_altitude,
                yaw=0.0,
            ),
            initial_policy_metadata=initial_policy_metadata,
            search_grid=search_grid,
            belief_updater=BayesianBeliefUpdater(sensor_model),
        )

        # Candidate policies must replan after every Bayesian update.
        if policy_name != 'coverage':
            return await self._simulate_online_search(
                robot_id=robot_id,
                robot_type=robot_type,
                robot_pos=robot_pos,
                params=params,
                update_callback=update_callback,
                interrupt_check=interrupt_check,
                execution_time=float(exec_time),
                search_task=search_task,
                search_grid=search_grid,
                camera_model=camera_model,
                search_session=search_session,
                prior_projection=prior_projection,
            )

        coverage_viewpoints = search_session.remaining_plan()
        base_waypoints = [[viewpoint.x, viewpoint.y] for viewpoint in coverage_viewpoints]
        if not base_waypoints:
            logger.error(f"Failed to generate base waypoints for {search_type}")
            return {'success': False, 'outcome': 'waypoint_generation_failed'}

        # 2. Build the final waypoint sequence by mode
        final_waypoints: List[List[float]] = []
        if search_type == 'patrol':
            # Patrol mode: build a back-and-forth sequence
            loops = int(params.get('loops', params.get('repetitions', 3)))
            loops = max(1, loops)
            for k in range(loops):
                # Alternate between forward and reverse sequences
                seq = base_waypoints if (k % 2 == 0) else list(reversed(base_waypoints))
                if not final_waypoints:  # Add directly on the first pass
                    final_waypoints.extend(seq)
                else:
                    # Deduplicate: skip the first point if it matches the previous pass's last point
                    if np.allclose(final_waypoints[-1], seq[0], atol=1e-6):
                        final_waypoints.extend(seq[1:])
                    else:
                        final_waypoints.extend(seq)
        else:  # 'search' mode
            # Search mode: use base waypoints directly
            final_waypoints = base_waypoints

        # 3. Set update callback
        def search_update_callback(motion_data: Dict):
            motion_data['robot_id'] = robot_id
            if 'phase' not in motion_data:
                motion_data['phase'] = 'patrolling' if search_type == 'patrol' else 'searching'
            update_callback(motion_data)

            # Sync carried object positions
            if 'position' in motion_data:
                carried_ids = self.scene_graph.get_neighbors_by_relation(robot_id, 'carrying')
                for carried_id in carried_ids:
                    self._entity_motion_states[carried_id] = motion_data['position']
                    prop = self.scene_graph.get_prop(carried_id)
                    prop_label = prop.get('properties', {}).get('label', '') if prop else ''
                    if self._can_viz():
                        self.visualizer.update_entity_position(
                            entity_id=carried_id,
                            position=motion_data['position'],
                            additional_info={'label': prop_label, 'carried': True}
                        )

        # 4. Call the underlying simulator
        result = await self.motion_simulator.simulate_search(
            robot_type=robot_type,
            start_pos=robot_pos,
            search_waypoints=final_waypoints,  
            update_callback=search_update_callback,
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )

        if result['success']:
            self._entity_motion_states[robot_id] = result['final_position']

        # Patrol keeps its legacy motion-completion semantics. Active search is
        # closed through SearchObservation -> SearchState -> SearchOutcome.
        if search_type == 'patrol':
            return result

        if not result['success']:
            reason = str(result.get('reason') or 'search motion failed')
            outcome = (
                search_session.abort(reason)
                if reason == 'interrupted'
                else search_session.fail(reason)
            )
            result.update(outcome.to_platform_result())
            return result

        route_size = len(coverage_viewpoints)
        time_per_viewpoint = float(result.get('actual_time', 0.0)) / route_size
        distance_per_viewpoint = float(result.get('path_length', 0.0)) / route_size
        energy_per_viewpoint = float(result.get('energy_used', 0.0)) / route_size

        simulated_confidence = max(
            0.0,
            min(1.0, float(params.get('simulated_detection_confidence', 1.0))),
        )

        for index in range(route_size):
            viewpoint = search_session.next_viewpoint()
            if viewpoint is None:
                break
            visible_cells = search_grid.cells_within_radius(
                viewpoint.x,
                viewpoint.y,
                camera_model.resolved_footprint_radius_m,
            )
            search_session.record_observation(SearchObservation(
                viewpoint=viewpoint,
                timestamp_s=(index + 1) * time_per_viewpoint,
                detections=self._simulated_viewpoint_detections(
                    params,
                    search_task,
                    search_grid,
                    tuple(cell.cell_id for cell in visible_cells),
                    simulated_confidence,
                ),
                visible_cell_ids=tuple(cell.cell_id for cell in visible_cells),
                travel_time_s=time_per_viewpoint,
                travel_distance_m=distance_per_viewpoint,
                energy_used=energy_per_viewpoint,
                sensor_metadata={
                    'source': 'semantic_simulator',
                    'grid_resolution_m': search_grid.resolution_m,
                    'footprint_radius_m': camera_model.resolved_footprint_radius_m,
                },
            ))
            if search_session.completed:
                break

        if not search_session.completed:
            search_session.next_viewpoint()
        assert search_session.outcome is not None
        result.update(search_session.outcome.to_platform_result())
        self._attach_search_intelligence_result(
            result,
            search_session,
            prior_projection,
        )
        result['motion_completed'] = True
        return result

    async def _simulate_online_search(
        self,
        *,
        robot_id: int,
        robot_type: str,
        robot_pos: List[float],
        params: Dict,
        update_callback: Callable,
        interrupt_check: Callable,
        execution_time: float,
        search_task: SearchTask,
        search_grid: SearchGrid,
        camera_model: CandidateViewpointGenerator,
        search_session: SearchSession,
        prior_projection: Optional[Any],
    ) -> Dict:
        """Execute one candidate at a time so the policy sees each posterior."""
        current_position = list(robot_pos)
        cumulative_time = 0.0
        cumulative_distance = 0.0
        cumulative_energy = 0.0
        max_steps = search_task.budget.max_viewpoints or len(
            search_session.remaining_plan()
        )
        step_execution_time = execution_time / max(1, max_steps)
        simulated_confidence = max(
            0.0,
            min(1.0, float(params.get('simulated_detection_confidence', 1.0))),
        )

        def online_update_callback(motion_data: Dict):
            motion_data['robot_id'] = robot_id
            motion_data.setdefault('phase', 'searching')
            update_callback(motion_data)

            if 'position' in motion_data:
                carried_ids = self.scene_graph.get_neighbors_by_relation(
                    robot_id, 'carrying'
                )
                for carried_id in carried_ids:
                    self._entity_motion_states[carried_id] = motion_data['position']
                    prop = self.scene_graph.get_prop(carried_id)
                    prop_label = (
                        prop.get('properties', {}).get('label', '') if prop else ''
                    )
                    if self._can_viz():
                        self.visualizer.update_entity_position(
                            entity_id=carried_id,
                            position=motion_data['position'],
                            additional_info={
                                'label': prop_label,
                                'carried': True,
                            },
                        )

        last_motion_result: Dict[str, Any] = {}
        while not search_session.completed:
            viewpoint = search_session.next_viewpoint()
            if viewpoint is None:
                break
            waypoint = [viewpoint.x, viewpoint.y]
            if len(current_position) > 2:
                waypoint.append(viewpoint.z)

            motion_result = await self.motion_simulator.simulate_search(
                robot_type=robot_type,
                start_pos=current_position,
                search_waypoints=[waypoint],
                update_callback=online_update_callback,
                interrupt_check=interrupt_check,
                execution_time=step_execution_time,
            )
            last_motion_result = motion_result
            cumulative_time += float(motion_result.get('actual_time', 0.0))
            cumulative_distance += float(motion_result.get('path_length', 0.0))
            cumulative_energy += float(motion_result.get('energy_used', 0.0))
            current_position = list(
                motion_result.get('final_position', current_position)
            )

            if not motion_result.get('success'):
                reason = str(motion_result.get('reason') or 'search motion failed')
                outcome = (
                    search_session.abort(reason)
                    if reason == 'interrupted'
                    else search_session.fail(reason)
                )
                result = dict(motion_result)
                result.update(outcome.to_platform_result())
                self._attach_search_intelligence_result(
                    result, search_session, prior_projection
                )
                result['motion_completed'] = False
                return result

            visible_cells = search_grid.cells_within_radius(
                viewpoint.x,
                viewpoint.y,
                camera_model.resolved_footprint_radius_m,
            )
            visible_cell_ids = tuple(cell.cell_id for cell in visible_cells)
            search_session.record_observation(SearchObservation(
                viewpoint=viewpoint,
                timestamp_s=cumulative_time,
                detections=self._simulated_viewpoint_detections(
                    params,
                    search_task,
                    search_grid,
                    visible_cell_ids,
                    simulated_confidence,
                ),
                visible_cell_ids=visible_cell_ids,
                travel_time_s=float(motion_result.get('actual_time', 0.0)),
                travel_distance_m=float(motion_result.get('path_length', 0.0)),
                energy_used=float(motion_result.get('energy_used', 0.0)),
                sensor_metadata={
                    'source': 'semantic_simulator',
                    'grid_resolution_m': search_grid.resolution_m,
                    'footprint_radius_m': camera_model.resolved_footprint_radius_m,
                },
            ))

        if not search_session.completed:
            search_session.next_viewpoint()
        assert search_session.outcome is not None
        result = dict(last_motion_result)
        result.update({
            'final_position': current_position,
            'actual_time': cumulative_time,
            'path_length': cumulative_distance,
            'energy_used': cumulative_energy,
            'avg_velocity': (
                cumulative_distance / cumulative_time if cumulative_time > 0 else 0.0
            ),
            'area_covered': all(
                cell.cell_id in search_session.state.observed_cell_quality
                for cell in search_grid.searchable_cells
            ),
        })
        result.update(search_session.outcome.to_platform_result())
        self._attach_search_intelligence_result(
            result, search_session, prior_projection
        )
        result['motion_completed'] = True
        if current_position:
            self._entity_motion_states[robot_id] = current_position
        return result

    def _simulated_viewpoint_detections(
        self,
        params: Dict,
        search_task: SearchTask,
        search_grid: SearchGrid,
        visible_cell_ids: tuple,
        confidence: float,
    ) -> tuple:
        """Keep simulator ground truth behind the observation adapter."""
        raw_target_ids = params.get('target_ids')
        if raw_target_ids is None:
            raw_target_ids = ()
        if not isinstance(raw_target_ids, (list, tuple, set)):
            raw_target_ids = (raw_target_ids,)
        visible = set(visible_cell_ids)
        detections = []
        for target_id in raw_target_ids:
            target_position = self.get_entity_position(target_id)
            if target_position is None or len(target_position) < 2:
                continue
            target_cell = search_grid.cell_at(
                float(target_position[0]), float(target_position[1])
            )
            if target_cell is None or target_cell.cell_id not in visible:
                continue
            estimated_position = (
                float(target_position[0]),
                float(target_position[1]),
                float(target_position[2]) if len(target_position) > 2 else 0.0,
            )
            detections.append(TargetDetection(
                label=search_task.target.query,
                confidence=confidence,
                estimated_position=estimated_position,
                entity_id=str(target_id),
            ))
        return tuple(detections)

    @staticmethod
    def _attach_search_intelligence_result(
        result: Dict,
        search_session: SearchSession,
        prior_projection: Optional[Any],
    ) -> None:
        result['policy_trace'] = list(search_session.policy_decisions)
        result['belief_trace'] = [
            {
                'update_index': update.posterior.update_index,
                'evidence_type': update.evidence_type,
                'evidence_cell_ids': update.evidence_cell_ids,
                'prior_entropy_nats': update.prior_entropy_nats,
                'posterior_entropy_nats': update.posterior_entropy_nats,
                'entropy_reduction_nats': update.entropy_reduction_nats,
                'kl_divergence_nats': update.kl_divergence_nats,
                'most_likely_cell_id': update.posterior.most_likely_cell_id,
                'maximum_probability': update.posterior.maximum_probability,
            }
            for update in search_session.belief_updates
        ]
        if prior_projection is not None:
            result['search_prior_diagnostics'] = {
                'confidence': prior_projection.confidence,
                'matched_labels': prior_projection.matched_labels,
                'unmatched_labels': prior_projection.unmatched_labels,
            }

    async def _simulate_take_off(
        self, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulate drone takeoff."""
        if robot_type not in ['UAV', 'FW_UAV']:
            return {'success': False, 'outcome': 'not_applicable'}

        target_altitude = params.get('target_altitude', 50.0)
        exec_time = params.get('execution_time', 2.0)

        result = await self.motion_simulator.simulate_take_off(
            drone_pos=robot_pos,
            target_altitude=target_altitude,
            update_callback=update_callback,
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )

        return result

    async def _simulate_land(
        self, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulate drone landing."""
        if robot_type not in ['UAV', 'FW_UAV']:
            return {'success': False, 'outcome': 'not_applicable'}

        current_altitude = params.get('current_altitude', 50.0)
        exec_time = params.get('execution_time', 2.0)

        result = await self.motion_simulator.simulate_land(
            drone_pos=robot_pos,
            current_altitude=current_altitude,
            update_callback=update_callback,
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )

        return result

    async def _simulate_return_home(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulate UAV/FW_UAV return to base (home_position comes from params)."""
        home_pos = params.get('home_position', [0, 0])
        exec_time = params.get('execution_time', 5.0)

        obstacles = self._get_obstacles_info(robot_id, None)

        # Navigation phase
        nav_result = await self.motion_simulator.simulate_navigate(
            robot_type=robot_type,
            start_pos=robot_pos,
            end_pos=home_pos,
            obstacles=obstacles,
            update_callback=lambda data: update_callback({**data, 'phase': 'returning'}),
            interrupt_check=interrupt_check,
            execution_time=exec_time * 0.7
        )

        if not nav_result['success']:
            return nav_result

        # Landing phase
        land_result = await self.motion_simulator.simulate_land(
            drone_pos=home_pos,
            current_altitude=params.get('current_altitude', 50.0),
            update_callback=lambda data: update_callback({**data, 'phase': 'landing'}),
            interrupt_check=interrupt_check,
            execution_time=exec_time * 0.3
        )

        if land_result['success']:
            self._entity_motion_states[robot_id] = home_pos

        return {
            'success': land_result['success'],
            'final_position': home_pos,
            'outcome': 'at_home_base' if land_result['success'] else 'landing_failed'
        }

    async def _simulate_follow(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulate target following (target comes from params.object_id)."""

        target_id = params.get("object_id")
        if not target_id:
            return {'success': False, 'outcome': 'no_target_specified'}

        target_pos = self.get_entity_position(target_id)
        if not target_pos:
            return {'success': False, 'outcome': 'target_not_found'}

        exec_time = params.get('execution_time', 10.0)
        following_distance = params.get('following_distance', 5.0)

        def follow_update_callback(motion_data: Dict):
            current_target_pos = self.get_entity_position(target_id)
            if current_target_pos:
                motion_data['target_position'] = current_target_pos
            update_callback(motion_data)

            if 'position' in motion_data:
                self._entity_motion_states[robot_id] = motion_data['position']

        result = await self.motion_simulator.simulate_follow(
            robot_type=robot_type,
            robot_pos=robot_pos,
            target_id=target_id,
            get_target_position=lambda: self.get_entity_position(target_id),
            following_distance=following_distance,
            update_callback=follow_update_callback,
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )

        if result['success']:
            self._entity_motion_states[robot_id] = result['final_position']

        return result
    
    async def _simulate_broadcast(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """
        Simulate the broadcast skill:
        - params:
            message: text to broadcast
            execution_time: total duration, default 3.0s (adjustable if needed)
        """
        message = params.get('message', '')
        exec_time = params.get('execution_time', 3.0)

        # Wrap the callback so all updates use the visualization path
        def broadcast_update_callback(motion_data: Dict):
            # Pass message through unchanged for HUD display
            motion_data.setdefault('message', message)
            self._update_skill_visualization(
                robot_id, 
                self.scene_graph.get_robot(robot_id).get('properties', {}).get('label', ''), 
                SkillName.BROADCAST.value, 
                motion_data
            )

        # Call the underlying simulator directly
        result = await self.motion_simulator.simulate_broadcast(
            robot_type=robot_type,
            robot_pos=robot_pos,
            message=message,
            update_callback=broadcast_update_callback,
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )

        # Broadcast does not change position, but can mark completion after success
        if result.get('success'):
            # Keep current position and mark skill_ended
            if self._can_viz():
                self.visualizer.update_entity_position(
                    entity_id=robot_id,
                    position=self.get_entity_position(robot_id) or robot_pos,
                    additional_info={
                        'label': self.scene_graph.get_robot(robot_id).get('properties', {}).get('label', ''),
                        'skill_ended': True
                    }
                )

        return result
    
    async def _simulate_place(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulates place, load, or unload based on params."""
        obj_to_place_id = params.get('object_id')
        if obj_to_place_id is None:
            return {'success': False, 'outcome': 'missing_object_id'}

        # Parse surface_target: supports structured format and legacy string format
        surface_target = params.get('surface_target', {})
        if isinstance(surface_target, dict):
            surface_class = surface_target.get('class', '')
        else:
            surface_class = 'robot' if surface_target == 'ugv' else ('ground' if surface_target == 'ground' else '')

        weight = params.get('weight', 5.0)
        exec_time = params.get('execution_time', 2.5)

        # Assumed to be with the robot at the start of the action
        object_start_pos = self.get_entity_position(robot_id)
        if not object_start_pos:
            return {'success': False, 'outcome': 'robot_position_unknown'}

        if surface_class == 'robot':
            # Load onto a carrier (UGV)
            carrier_id = params.get('carrier_id')
            if carrier_id is None:
                return {'success': False, 'outcome': 'missing_carrier_id'}
            carrier_pos = self.get_entity_position(carrier_id)
            if not carrier_pos:
                return {'success': False, 'outcome': 'carrier_position_unknown'}
            if self._can_viz():
                self.visualizer.update_edge_state(carrier_id, obj_to_place_id, 'carrying', 'highlight')
            return await self.motion_simulator.simulate_load_object(
                robot_type=robot_type, robot_pos=robot_pos,
                object_pos=carrier_pos,
                object_weight=weight, update_callback=update_callback,
                interrupt_check=interrupt_check, execution_time=exec_time
            )
        elif surface_class == 'ground':
            # Unload to the ground
            carried_ids = self.scene_graph.get_neighbors_by_relation(robot_id, 'carrying')
            result = await self.motion_simulator.simulate_unload_object(
                robot_type=robot_type, robot_pos=robot_pos,
                target_pos=robot_pos,
                object_weight=weight, update_callback=update_callback,
                interrupt_check=interrupt_check, execution_time=exec_time
            )
            if result.get('success'):
                for carried_id in carried_ids:
                    if self._can_viz():
                        carrier_id = params.get('carrier_id')
                        if carrier_id:
                            self.visualizer.update_edge_state(carrier_id, carried_id, 'carrying', 'normal')
                        prop = self.scene_graph.get_prop(carried_id)
                        prop_label = prop.get('properties', {}).get('label', '') if prop else ''
                        self.visualizer.update_entity_position(
                            entity_id=carried_id, position=robot_pos, additional_info={'label': prop_label}
                        )
            return result
        else:
            # Place on another object's surface
            surface_id = params.get('surface_id')
            if surface_id is None:
                return {'success': False, 'outcome': 'missing_surface_object_id'}
            
            surface_pos = self.get_entity_position(surface_id)
            if not surface_pos:
                return {'success': False, 'outcome': 'surface_position_unknown'}

            def place_update_callback(motion_data: Dict):
                update_callback(motion_data)
                if 'object_position' in motion_data and self._can_viz():
                    prop = self.scene_graph.get_prop(obj_to_place_id)
                    prop_label = prop.get('properties', {}).get('label', '') if prop else ''
                    self.visualizer.update_entity_position(
                        entity_id=obj_to_place_id,
                        position=motion_data['object_position'],
                        additional_info={'label': prop_label, 'status': 'placing'}
                    )
            return await self.motion_simulator.simulate_place(
                robot_type=robot_type, robot_pos=robot_pos,
                object_start_pos=object_start_pos, surface_pos=surface_pos,
                object_weight=weight, update_callback=place_update_callback,
                interrupt_check=interrupt_check, execution_time=exec_time
            )

    async def _simulate_handle_hazard(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulate handling a hazard."""
        hazard_id = params.get('object_id')
        if not hazard_id:
            return {'success': False, 'outcome': 'no_hazard_specified'}
        
        hazard_pos = self.get_entity_position(hazard_id)
        if not hazard_pos:
            return {'success': False, 'outcome': 'hazard_position_unknown'}

        exec_time = params.get('execution_time', 4.0)
        
        result = await self.motion_simulator.simulate_handle_hazard(
            robot_type=robot_type,
            robot_pos=robot_pos,
            hazard_pos=hazard_pos,
            update_callback=update_callback,
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )
        return result

    async def _simulate_guide(
        self, robot_id: int, robot_type: str, robot_pos: List[float],
        params: Dict, update_callback: Callable, interrupt_check: Callable
    ) -> Dict:
        """Simulate guiding an entity to a location."""
        # Get guided entity ID
        guided_id = params.get('object_id')
        if guided_id is None:
            return {'success': False, 'outcome': 'missing_guide_target'}
        
        # Get destination position
        dest_pos = None
        # 1) Explicit coordinates
        d = params.get('dest')
        if isinstance(d, dict) and isinstance(d.get('x'), (int, float)) and isinstance(d.get('y'), (int, float)):
            dest_pos = [float(d['x']), float(d['y'])]
        # 2) Derive from area
        if dest_pos is None and isinstance(params.get('area'), dict):
            a = params['area']; k = a.get('kind')
            if k == 'point' and a.get('coords'):
                dest_pos = [float(a['coords'][0][0]), float(a['coords'][0][1])]
            elif k == 'circle' and a.get('center'):
                dest_pos = [float(a['center'][0]), float(a['center'][1])]
            elif k == 'area' and a.get('coords'):
                xs = [c[0] for c in a['coords']]; ys = [c[1] for c in a['coords']]
                dest_pos = [sum(xs)/len(xs), sum(ys)/len(ys)]
        # 3) Fall back to destination_id
        destination_id = params.get('destination_id')
        if dest_pos is None and destination_id is not None:
            pos = self.get_entity_position(destination_id)
            if pos:
                dest_pos = pos
        if dest_pos is None:
            return {'success': False, 'outcome': 'destination_position_unknown'}
            
        exec_time = params.get('execution_time', 5.0)
        obstacles = self._get_obstacles_info(robot_id, destination_id)

        # Custom callback to move both the robot and the guided entity
        def guide_update_callback(motion_data: Dict):
            update_callback(motion_data)
            
            if 'position' in motion_data:
                robot_current_pos = np.array(motion_data['position'])
                # Guided entity follows slightly behind
                direction = motion_data.get('direction', np.array([1, 0]))
                follow_pos = robot_current_pos - direction * 1.0 # 1 meters behind
                guided_node = self.scene_graph.get_node_by_id(guided_id)
                guided_label = guided_node.get('properties', {}).get('label', '') if guided_node else ''
                self._entity_motion_states[guided_id] = follow_pos.tolist()
                if self._can_viz():
                    self.visualizer.update_entity_position(
                        entity_id=guided_id,
                        position=follow_pos.tolist(),
                        additional_info={'label': guided_label, 'status': 'following'}
                    )

        # The core motion is a navigation for the robot
        result = await self.motion_simulator.simulate_guide(
            robot_type=robot_type,
            start_pos=robot_pos,
            end_pos=dest_pos,
            obstacles=obstacles,
            update_callback=guide_update_callback,
            interrupt_check=interrupt_check,
            execution_time=exec_time
        )

        if result['success']:
            self._entity_motion_states[robot_id] = result['final_position']
            self._entity_motion_states[guided_id] = result['final_position']

        return result

    def _update_skill_visualization(
        self, robot_id: int, robot_label: str,
        skill: str, motion_data: Dict
    ):
        """
        Update skill execution visualization.
        """
        display_info = {}

        # Update position in memory
        if 'position' in motion_data:
            self._entity_motion_states[robot_id] = motion_data['position']
            display_info['position'] = motion_data['position']
        else:
            if robot_id in self._entity_motion_states:
                display_info['position'] = self._entity_motion_states[robot_id]
            else:
                robot_pos = self.get_entity_position(robot_id)
                if robot_pos:
                    self._entity_motion_states[robot_id] = robot_pos
                    display_info['position'] = robot_pos
                else:
                    logger.warning(f"Cannot get position for robot {robot_id}, using default [0, 0]")
                    display_info['position'] = [0, 0]

        # Basic info
        display_info['label'] = robot_label

        # Skill name (short form)
        skill_names = {
            SkillName.NAVIGATE.value: "NAV",
            SkillName.TAKE_PHOTO.value: "PHOTO",
            SkillName.TAKE_OFF.value: "TAKEOFF",
            SkillName.LAND.value: "LAND",
            SkillName.SEARCH.value: "SEARCH",
            SkillName.RETURN_HOME.value: "RETURN",
            SkillName.FOLLOW.value: "FOLLOW",
            SkillName.BROADCAST.value: "BCAST",
            SkillName.PLACE.value: "PLACE",
            SkillName.HANDLE_HAZARD.value: "HAZARD",
            SkillName.GUIDE.value: "GUIDE",
        }
        display_info['skill'] = skill_names.get(skill, skill[:6].upper())

        # Phase
        if 'phase' in motion_data:
            display_info['phase'] = motion_data['phase'].upper()

        # Progress percentage
        if 'progress' in motion_data:
            display_info['progress'] = int(motion_data['progress'] * 100)

        # Speed (navigation only)
        if skill == SkillName.NAVIGATE.value and 'velocity' in motion_data:
            if motion_data['velocity'] > 0.1:
                display_info['velocity'] = round(motion_data['velocity'], 1)

        # Altitude (takeoff, landing, or return home)
        if skill in [SkillName.TAKE_OFF.value, SkillName.LAND.value, SkillName.RETURN_HOME.value]:
            if 'altitude' in motion_data:
                display_info['altitude'] = round(motion_data['altitude'], 1)

        # Follow info
        if skill == SkillName.FOLLOW.value:
            if 'target_distance' in motion_data:
                display_info['target_dist'] = round(motion_data['target_distance'], 1)
            if 'following_status' in motion_data:
                display_info['follow_status'] = motion_data['following_status']

        # Photo type
        if skill == SkillName.TAKE_PHOTO.value and 'photo_type' in motion_data:
            display_info['photo_type'] = motion_data['photo_type']
        
        # Broadcast content
        if skill == SkillName.BROADCAST.value and 'message' in motion_data:
            display_info['message'] = motion_data['message']

        # Update visualization
        if self._can_viz():
            self.visualizer.update_entity_position(
                entity_id=robot_id,
                position=display_info['position'],
                additional_info=display_info
            )

    async def simulate_autonomous_motion(
        self, entity_id: int, entity_type: str,
        entity_label: str, start_pos: List[float],
        end_pos: List[float], duration: float,
        reason: str,
        obstacles: List = None
    ) -> Dict:
        """
        Simulate autonomous entity motion (for example, target vehicle movement).
        """
        if self.is_entity_disabled(entity_id):
            logger.warning(f"Entity {entity_label} is disabled, cannot move")
            return {
                'success': False,
                'reason': 'entity_disabled',
                'final_position': start_pos
            }

        motion_task = {
            'entity_id': entity_id,
            'entity_label': entity_label,
            'entity_type': entity_type,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'duration': duration,
            'reason': reason
        }
        self._autonomous_motions_in_progress[entity_id] = motion_task

        def update_callback(motion_data: Dict):
            display_info = {
                'label': entity_label,
                'type': entity_type,
                'status': 'MOVING',
                'motion_type': motion_data.get('motion_type', 'autonomous'),
                'reason': reason,
                'progress': f"{int(motion_data.get('progress', 0) * 100)}%"
            }

            if 'position' in motion_data:
                self._entity_motion_states[entity_id] = motion_data['position']

            if self._can_viz():
                self.visualizer.update_entity_position(
                    entity_id=entity_id,
                    position=motion_data.get('position', [0, 0]),
                    additional_info=display_info
                )

        def interrupt_check():
            return False

        try:
            result = await self.motion_simulator.simulate_autonomous_movement(
                entity_id=entity_id,
                entity_type=entity_type,
                start_pos=start_pos,
                end_pos=end_pos,
                update_callback=update_callback,
                interrupt_check=interrupt_check,
                movement_time=duration,
                reason=reason,
                obstacles=obstacles
            )

            if result['success']:
                logger.info(f"Autonomous motion completed for {entity_label}")
                self._entity_motion_states[entity_id] = result['final_position']

                # Final state should also include the label
                final_display_info = {
                    'label': entity_label,
                    'type': entity_type,
                    'status': 'IDLE'
                }

                if self._can_viz():
                    self.visualizer.update_entity_position(
                        entity_id=entity_id,
                        position=result['final_position'],
                        additional_info=final_display_info
                    )

            return result

        except Exception as e:
            logger.error(f"Error during autonomous motion for {entity_label}: {e}")
            return {
                'success': False,
                'reason': 'simulation_error',
                'error': str(e)
            }
        finally:
            self._autonomous_motions_in_progress.pop(entity_id, None)

    def _update_autonomous_motion_visualization(
        self, entity_id: int, entity_label: str,
        entity_type: str, motion_data: Dict, reason: str
    ):
        """Update autonomous motion visualization."""
        display_info = {
            'label': entity_label,
            'type': entity_type,
            'status': 'MOVING',
            'motion_type': motion_data.get('motion_type', 'autonomous'),
            'reason': reason,
            'progress': f"{int(motion_data.get('progress', 0) * 100)}%"
        }

        if 'position' in motion_data:
            self._entity_motion_states[entity_id] = motion_data['position']

        if self._can_viz():
            self.visualizer.update_entity_position(
                entity_id=entity_id,
                position=motion_data.get('position', [0, 0]),
                additional_info=display_info
            )

    def set_entity_disabled(self, entity_id: int, entity_label: str, disabled: bool = True):
        """
        Set visualization for the entity disabled state.
        """
        if disabled:
            self.motion_simulator.entity_state_manager.set_entity_disabled(entity_id, True)

            if entity_id in self._autonomous_motions_in_progress:
                logger.info(f"Stopping ongoing motion for disabled entity: {entity_label}")

            current_pos = self.get_entity_position(entity_id)
            if current_pos:
                if self._can_viz():
                    self.visualizer.update_entity_position(
                        entity_id=entity_id,
                        position=current_pos,
                        additional_info={
                            'label': entity_label,
                            'status': 'ERROR',
                            'disabled': True,
                            'color': 'red',
                            'marker': 'X'
                        }
                    )

            logger.warning(f"Entity {entity_label} has been disabled")
        else:
            self.motion_simulator.entity_state_manager.set_entity_disabled(entity_id, False)

            current_pos = self.get_entity_position(entity_id)
            if current_pos:
                if self._can_viz():
                    self.visualizer.update_entity_position(
                        entity_id=entity_id,
                        position=current_pos,
                        additional_info={
                            'label': entity_label,
                            'status': 'IDLE',
                            'disabled': False,
                            'color': 'green',
                            'marker': 'O'
                        }
                    )

            logger.info(f"Entity {entity_label} has been recovered")

    def get_entity_position(self, entity_id: Optional[int]) -> Optional[List[float]]:
        """
        Get the entity's current position, preferring motion state.
        """
        if entity_id is None:
            return None

        if entity_id in self._entity_motion_states:
            return self._entity_motion_states[entity_id]

        for n in self.scene_graph._nodes:
            if str(n['id']) == str(entity_id):
                shape = n.get('shape', {})
                center = shape_center_point(shape)
                return center

        return None

    def is_entity_disabled(self, entity_id: int) -> bool:
        """
        Check whether the entity is disabled.
        """
        return self.motion_simulator.entity_state_manager.is_entity_disabled(entity_id)

    def _get_obstacles_info(self, robot_id: int, target_id: Optional[int]) -> List:
        """
        Get obstacle information along the path.
        """
        obstacles = []
        excluded_ids = set([robot_id])  # Always exclude the robot itself

        # Get the building ID where the robot is currently located
        robot_node = self.scene_graph.get_node_by_id(robot_id)
        if robot_node:
            robot_location = robot_node.get('properties', {}).get('location', {})
            if isinstance(robot_location, dict) and 'label' in robot_location:
                robot_location_label = robot_location['label']
                robot_location_id = self.label_to_id_map.get(robot_location_label, None)
                if robot_location_id:
                    excluded_ids.add(robot_location_id)
            elif isinstance(robot_location, (int, str)):
                excluded_ids.add(robot_location)

        # If the target is a building, exclude it as well
        target_node = self.scene_graph.get_node_by_id(target_id) if target_id is not None else None
        if target_node and target_node.get('properties', {}).get('category') == 'building':
            excluded_ids.add(target_id)

        # Get all nodes
        all_nodes = self.scene_graph.get_all_nodes()

        for node in all_nodes:
            node_id = node.get('id')

            if node_id in excluded_ids:
                continue

            category = node.get('properties', {}).get('category')
            if category not in ['building']:
                continue

            shape = node.get('shape')
            if shape:
                center = shape_center_point(shape)
                if center is not None:
                    position = [float(center[0]), float(center[1])]

                    from modules.platform.semantic_platform.skill_simulator import ObstacleInfo
                    obstacles.append(ObstacleInfo(
                        position=np.array(position),
                        shape=shape,
                        is_dynamic=False
                    ))

        logger.debug(f"Found {len(obstacles)} obstacles for robot {robot_id} navigating to {target_id}")
        return obstacles

    async def cleanup_skill_display(self, robot_id: int, delay: float = 2.0):
        """
        Clean up skill display.
        """
        await asyncio.sleep(delay)
        if not self._can_viz():
            return

        position = self.get_entity_position(robot_id)
        if position:
            robot = self.scene_graph.get_robot(robot_id)
            robot_label = robot.get('properties', {}).get('label', '') if robot else ''

            if not self.is_entity_disabled(robot_id):
                if self._can_viz():
                    self.visualizer.update_entity_position(
                        entity_id=robot_id,
                        position=position,
                        additional_info={
                            'label': robot_label,
                            'skill_ended': True
                        }
                    )

    async def cleanup_task_labels(self, skill: str, object_id: Optional[int], delay: float = 1.0):
        """
        Clean up task-related label display, keyed by object_id.
        """
        await asyncio.sleep(delay)
        if not self._can_viz():
            return

        # Clean up task labels for the target object
        if object_id:
            obj = self.scene_graph.get_node_by_id(object_id)
            obj_pos = self.get_entity_position(object_id)
            if obj and obj_pos:
                if self._can_viz():
                    self.visualizer.update_entity_position(
                        entity_id=object_id,
                        position=obj_pos,
                        additional_info={'task_ended': True}
                    )

        # Mark the target, if any, as navigation-ended after navigation completes
        if object_id and skill == SkillName.NAVIGATE.value:
            obj_pos = self.get_entity_position(object_id)
            if obj_pos:
                if self._can_viz():
                    self.visualizer.update_entity_position(
                        entity_id=object_id,
                        position=obj_pos,
                        additional_info={'navigation_ended': True}
                    )
