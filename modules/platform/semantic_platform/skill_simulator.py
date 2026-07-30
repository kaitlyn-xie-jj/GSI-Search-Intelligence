import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math
import logging
import heapq

from .utils.motion_math import (
    smooth_step, smooth_step_derivative,
    interpolate_position, calculate_path_length, heading, clamp01
)
from .utils.geom_path_utils import (
    line_intersects_obstacle, point_in_any_obstacle,
    generate_straight_path, generate_bezier_path,
    generate_smooth_path, dijkstra
)


logger = logging.getLogger(__name__)

@dataclass
class MotionState:
    """Motion state data class."""
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    acceleration: np.ndarray  # [ax, ay]
    orientation: float  # Orientation angle in radians
    timestamp: float

@dataclass
class ObstacleInfo:
    """Obstacle information."""
    position: np.ndarray
    shape: Dict[str, List[float]]  # min_corner, max_corner
    is_dynamic: bool = False
    velocity: Optional[np.ndarray] = None
    
    def __eq__(self, other):
        """Custom equality comparison to avoid numpy array comparison issues."""
        if not isinstance(other, ObstacleInfo):
            return False
        # Compare shape dictionary, which is hashable
        return self.shape == other.shape and self.is_dynamic == other.is_dynamic

class MotionType(Enum):
    """Motion type."""
    LINEAR = "linear"
    BEZIER = "bezier"
    SPLINE = "spline"
    DYNAMIC_AVOID = "dynamic_avoid"

class SkillMotionSimulator:
    """Skill motion simulator responsible for continuous motion during skill execution."""
    
    def __init__(self, update_interval: float = 0.1, fast_mode: bool = False):
        """
        Args:
            update_interval: Position update interval in seconds.
            fast_mode: Fast mode; skip step-by-step simulation and compute only key frames.
        """
        self.update_interval = update_interval
        self.fast_mode = fast_mode
        self.motion_profiles = self._init_motion_profiles()
        # Add position cache to reduce repeated updates
        self._last_positions = {}  # entity_id -> last_position
        
    def _init_motion_profiles(self) -> Dict[str, Dict]:
        """Initialize motion parameters for different robot types."""
        return {
            "UAV": {
                "max_velocity": 30.0,  # m/s (increased to 30 m/s)
                "max_acceleration": 10.0,  # m/s^2 (increased to 10 m/s^2)
                "max_turn_rate": math.pi / 2,  # rad/s
                "preferred_altitude": 50.0,  # m
                "motion_type": MotionType.BEZIER
            },
            "FW_UAV": {
                "max_velocity": 30.0,  # m/s (increased to 30 m/s)
                "max_acceleration": 10.0,  # m/s^2 (increased to 10 m/s^2)
                "max_turn_rate": math.pi / 2,  # rad/s
                "preferred_altitude": 50.0,  # m
                "motion_type": MotionType.BEZIER
            },
            "UGV": {
                "max_velocity": 20.0,  # m/s (increased to 20 m/s)
                "max_acceleration": 6.0,  # m/s^2 (increased to 6 m/s^2)
                "max_turn_rate": math.pi / 4,
                "preferred_altitude": 0.0,
                "motion_type": MotionType.DYNAMIC_AVOID
            },
            "Quadruped": {
                "max_velocity": 8.0,  # m/s (quadrupeds are slower but flexible)
                "max_acceleration": 4.0,  # m/s^2
                "max_turn_rate": math.pi / 3,
                "preferred_altitude": 0.0,
                "motion_type": MotionType.DYNAMIC_AVOID,
                "can_traverse_rough": True  # Can traverse rough terrain
            },
            "Humanoid": {
                "max_velocity": 3.0,  # m/s (humanoid walking speed)
                "max_acceleration": 2.0,  # m/s^2
                "max_turn_rate": math.pi / 2,
                "preferred_altitude": 0.0,
                "motion_type": MotionType.DYNAMIC_AVOID,
                "manipulation_range": 2.0  # Manipulation range
            }
        }
    
    async def simulate_navigate(
        self,
        robot_type: str,
        start_pos: List[float],
        end_pos: List[float],
        obstacles: List[ObstacleInfo],
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Simulate navigate skill execution and guarantee arrival within execution_time.
        
        Args:
            robot_type: Robot type.
            start_pos: Start position [x, y].
            end_pos: Target position [x, y].
            obstacles: Obstacle list.
            update_callback: Position update callback.
            interrupt_check: Interrupt check function.
            execution_time: Execution time; arrival must happen within this time.
            
        Returns:
            Execution result dictionary.
        """
        profile = self.motion_profiles.get(robot_type, self.motion_profiles["UGV"])
        
        # Generate path
        path = []
        for point in self._plan_path(start_pos, end_pos, obstacles, profile):
            if isinstance(point, np.ndarray):
                path.append(point)
            else:
                path.append(np.array(point))
        
        # Ensure path is not empty
        if not path:
            path = [np.array(start_pos), np.array(end_pos)]
        
        # Calculate total path length and each segment length
        segment_lengths = []
        total_distance = 0.0
        for i in range(len(path) - 1):
            length = np.linalg.norm(path[i + 1] - path[i])
            segment_lengths.append(length)
            total_distance += length
        
        # Calculate required time for each segment, allocated by length ratio
        segment_times = []
        num_segments = len(path) - 1
        if total_distance > 0:
            for length in segment_lengths:
                segment_time = (length / total_distance) * execution_time * 0.95  # Leave 5% margin
                segment_times.append(segment_time)
        else:
            # If total distance is 0 but there are multiple segments, split time evenly
            if num_segments > 1e-6:
                time_per_segment = (execution_time * 0.95) / num_segments
                segment_times = [time_per_segment] * num_segments
            else:
                segment_times = []
        
        logger.debug(f"Navigation plan: total_distance={total_distance:.1f}m, "
                    f"execution_time={execution_time:.1f}s, "
                    f"required_avg_speed={total_distance/execution_time:.1f}m/s")
        
        # ===== Unified step-by-step simulation shared by standard and fast modes =====
        # fast_mode affects only sleep time, not calculation logic
        # Start motion simulation
        current_pos = np.array(start_pos)
        elapsed_time = 0.0
        path_index = 0
        
        while elapsed_time < execution_time and path_index < len(path) - 1:
            # Check interrupt
            if interrupt_check():
                return {
                    "success": False,
                    "final_position": current_pos.tolist(),
                    "reason": "interrupted"
                }
            
            # Current segment start and end
            segment_start = path[path_index]
            segment_end = path[path_index + 1]
            segment_time = segment_times[path_index]
            
            # Calculate elapsed time in current segment
            segment_elapsed = 0.0
            
            # Interpolate within current segment
            while segment_elapsed < segment_time and elapsed_time < execution_time:
                # Calculate current segment progress, from 0 to 1
                segment_progress = segment_elapsed / segment_time if segment_time > 0 else 1.0
                
                # Calculate current position with linear interpolation
                current_pos = segment_start + (segment_end - segment_start) * segment_progress
                
                # Calculate instantaneous velocity for display
                if segment_time > 0:
                    instant_velocity = np.linalg.norm(segment_end - segment_start) / segment_time
                else:
                    instant_velocity = 0.0
                
                # Calculate overall progress
                distance_traveled = sum(segment_lengths[:path_index]) + segment_lengths[path_index] * segment_progress
                overall_progress = distance_traveled / total_distance if total_distance > 0 else 1.0
                
                # Orientation
                direction = segment_end - segment_start
                if np.linalg.norm(direction) > 0.01:
                    orientation = math.atan2(direction[1], direction[0])
                else:
                    orientation = 0.0
                
                # Update callback only when position changes significantly
                position_changed = True
                if hasattr(self, '_last_positions'):
                    last_pos = self._last_positions.get(robot_type)
                    if last_pos is not None:
                        distance_moved = np.linalg.norm(current_pos - np.array(last_pos))
                        # Skip update if movement is less than 1 meter, unless this is a key frame
                        if distance_moved < 1.0 and segment_progress not in [0.0, 1.0]:
                            position_changed = False
                
                if position_changed:
                    # Callback update position
                    update_callback({
                        "position": current_pos.tolist(),
                        "velocity": instant_velocity,
                        "orientation": orientation,
                        "progress": min(overall_progress, elapsed_time / execution_time),
                        "path_index": path_index,
                        "total_points": len(path),
                        "time_elapsed": elapsed_time,
                        "time_total": execution_time
                    })
                    
                    # Record last position
                    if hasattr(self, '_last_positions'):
                        self._last_positions[robot_type] = current_pos.tolist()
                
                # Wait for next update cycle
                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                segment_elapsed += self.update_interval
                elapsed_time += self.update_interval
            
            # Move to next segment
            path_index += 1
        
        # Ensure final arrival at target position
        final_position = end_pos
        
        # Final update to ensure position and progress are both 100%
        update_callback({
            "position": final_position,
            "velocity": 0.0,
            "orientation": math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0]),
            "progress": 1.0,
            "path_index": len(path) - 1,
            "total_points": len(path),
            "time_elapsed": elapsed_time,
            "time_total": execution_time
        })
        
        return {
            "success": True,
            "final_position": final_position,
            "actual_time": elapsed_time,
            "path_length": total_distance,
            "avg_velocity": total_distance / elapsed_time if elapsed_time > 0 else 0
        }
    
    async def simulate_load_object(
        self,
        robot_type: str,
        robot_pos: List[float],
        object_pos: List[float],
        object_weight: float,
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Simulate the load-object skill.
        
        Includes: approach object -> align -> load action.
        """
        phases = [
            {"name": "approach", "duration_ratio": 0.3},
            {"name": "align", "duration_ratio": 0.2},
            {"name": "load", "duration_ratio": 0.5}
        ]
        
        elapsed_time = 0.0
        
        for phase in phases:
            phase_duration = execution_time * phase["duration_ratio"]
            phase_elapsed = 0.0
            
            while phase_elapsed < phase_duration:
                if interrupt_check():
                    return {
                        "success": False,
                        "phase": phase["name"],
                        "reason": "interrupted"
                    }
                
                progress = phase_elapsed / phase_duration
                
                if phase["name"] == "approach":
                    # Approach object
                    current_pos = interpolate_position(
                        robot_pos, object_pos, progress * 0.8  # Approach to 80% of the distance
                    )
                    update_callback({
                        "phase": "approach",
                        "position": current_pos,
                        "progress": progress
                    })
                
                elif phase["name"] == "align":
                    # Align action
                    update_callback({
                        "phase": "align",
                        "gripper_angle": progress * 90,  # Simulate gripper opening
                        "progress": progress
                    })
                
                elif phase["name"] == "load":
                    # Load action
                    lift_height = progress * 0.5  # Lift 0.5 meters
                    update_callback({
                        "phase": "load",
                        "object_height": lift_height,
                        "load_force": object_weight * 9.8,
                        "progress": progress
                    })
                
                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                phase_elapsed += self.update_interval
                elapsed_time += self.update_interval
        
        return {
            "success": True,
            "actual_time": elapsed_time,
            "loaded_weight": object_weight
        }
    
    async def simulate_unload_object(
        self,
        robot_type: str,
        robot_pos: List[float],
        target_pos: List[float],
        object_weight: float,
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """Simulate the unload-object skill."""
        phases = [
            {"name": "position", "duration_ratio": 0.3},
            {"name": "lower", "duration_ratio": 0.4},
            {"name": "release", "duration_ratio": 0.3}
        ]
        
        elapsed_time = 0.0
        
        for phase in phases:
            phase_duration = execution_time * phase["duration_ratio"]
            phase_elapsed = 0.0
            
            while phase_elapsed < phase_duration:
                if interrupt_check():
                    return {
                        "success": False,
                        "phase": phase["name"],
                        "reason": "interrupted"
                    }
                
                progress = phase_elapsed / phase_duration
                
                if phase["name"] == "position":
                    # Position at unload location
                    update_callback({
                        "phase": "position",
                        "alignment_error": (1 - progress) * 0.1,  # Alignment error gradually decreases
                        "progress": progress
                    })
                
                elif phase["name"] == "lower":
                    # Lower object
                    current_height = 0.5 * (1 - progress)  # Lower from 0.5 meters to 0
                    update_callback({
                        "phase": "lower",
                        "object_height": current_height,
                        "progress": progress
                    })
                
                elif phase["name"] == "release":
                    # Release object
                    gripper_angle = 90 * (1 - progress)  # Gripper gradually closes
                    update_callback({
                        "phase": "release",
                        "gripper_angle": gripper_angle,
                        "progress": progress
                    })
                
                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                phase_elapsed += self.update_interval
                elapsed_time += self.update_interval
        
        return {
            "success": True,
            "actual_time": elapsed_time,
            "unloaded_position": target_pos
        }
    
    async def simulate_take_photo(
        self,
        robot_type: str,
        robot_pos: List[float],
        target_pos: List[float],
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """Simulate the take-photo skill."""
        phases = [
            {"name": "aim", "duration_ratio": 0.4},
            {"name": "focus", "duration_ratio": 0.3},
            {"name": "capture", "duration_ratio": 0.3}
        ]
        
        elapsed_time = 0.0
        
        # Calculate shooting angle
        direction = np.array(target_pos) - np.array(robot_pos)
        target_angle = math.atan2(direction[1], direction[0])
        
        for phase in phases:
            phase_duration = execution_time * phase["duration_ratio"]
            phase_elapsed = 0.0
            
            while phase_elapsed < phase_duration:
                if interrupt_check():
                    return {
                        "success": False,
                        "phase": phase["name"],
                        "reason": "interrupted"
                    }
                
                progress = phase_elapsed / phase_duration
                
                if phase["name"] == "aim":
                    # Aim at target
                    current_angle = target_angle * progress
                    update_callback({
                        "phase": "aim",
                        "camera_angle": math.degrees(current_angle),
                        "progress": progress
                    })
                
                elif phase["name"] == "focus":
                    # Focus
                    focus_distance = np.linalg.norm(direction)
                    update_callback({
                        "phase": "focus",
                        "focus_distance": focus_distance,
                        "sharpness": progress,
                        "progress": progress
                    })
                
                elif phase["name"] == "capture":
                    # Capture
                    update_callback({
                        "phase": "capture",
                        "exposure": progress,
                        "progress": progress
                    })
                
                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                phase_elapsed += self.update_interval
                elapsed_time += self.update_interval
        
        return {
            "success": True,
            "actual_time": elapsed_time,
            "photo_metadata": {
                "angle": math.degrees(target_angle),
                "distance": np.linalg.norm(direction)
            }
        }
    
    async def simulate_take_off(
        self,
        drone_pos: List[float],
        target_altitude: float,
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """Simulate drone takeoff."""
        elapsed_time = 0.0
        current_altitude = 0.0
        
        # Takeoff phase: accelerated climb
        while elapsed_time < execution_time:
            if interrupt_check():
                return {
                    "success": False,
                    "final_altitude": current_altitude,
                    "reason": "interrupted"
                }
            
            progress = elapsed_time / execution_time
            
            # Use S-curve for smooth takeoff
            smooth_progress = smooth_step(progress)
            current_altitude = target_altitude * smooth_progress
            
            # Simulate rotor speed
            rotor_speed = min(progress * 1.2, 1.0) * 100  # Percentage
            
            update_callback({
                "altitude": current_altitude,
                "vertical_speed": (target_altitude / execution_time) * smooth_step_derivative(progress),
                "rotor_speed": rotor_speed,
                "progress": progress
            })
            
            if self.fast_mode:
                await asyncio.sleep(0)  # Almost no delay, but yields control
            else:
                await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
            elapsed_time += self.update_interval
        
        return {
            "success": True,
            "final_altitude": target_altitude,
            "actual_time": elapsed_time
        }
    
    async def simulate_land(
        self,
        drone_pos: List[float],
        current_altitude: float,
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """Simulate drone landing."""
        elapsed_time = 0.0
        altitude = current_altitude
        
        while elapsed_time < execution_time:
            if interrupt_check():
                return {
                    "success": False,
                    "final_altitude": altitude,
                    "reason": "interrupted"
                }
            
            progress = elapsed_time / execution_time
            
            # Use reverse S-curve for smooth landing
            smooth_progress = smooth_step(1 - progress)
            altitude = current_altitude * smooth_progress
            
            # Slow down during landing
            rotor_speed = smooth_progress * 100
            
            update_callback({
                "altitude": altitude,
                "vertical_speed": -(current_altitude / execution_time) * smooth_step_derivative(1 - progress),
                "rotor_speed": rotor_speed,
                "progress": progress
            })
            
            if self.fast_mode:
                await asyncio.sleep(0)  # Almost no delay, but yields control
            else:
                await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
            elapsed_time += self.update_interval
        
        return {
            "success": True,
            "final_altitude": 0.0,
            "actual_time": elapsed_time
        }
    
    async def simulate_search(
        self,
        robot_type: str,
        start_pos: List[float],
        search_waypoints: List[List[float]],
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Simulate search skill execution.
        """
        profile = self.motion_profiles.get(robot_type, self.motion_profiles["UAV"])
        
        if not search_waypoints:
            return {
                "success": True,
                "final_position": start_pos,
                "actual_time": 0.0,
                "path_length": 0.0,
                "avg_velocity": 0.0
            }
        
        logger.info(f"Starting search simulation with {len(search_waypoints)} waypoints, "
                f"execution_time={execution_time:.1f}s")
        
        # Build full path: start point + all waypoints
        path = [np.array(start_pos)]
        for wp in search_waypoints:
            path.append(np.array(wp))
        
        # Calculate total path length and each segment length
        segment_lengths = []
        total_distance = 0.0
        for i in range(len(path) - 1):
            length = np.linalg.norm(path[i + 1] - path[i])
            segment_lengths.append(length)
            total_distance += length
        
        # Allocate time by path length ratio
        segment_times = []
        num_segments = len(path) - 1
        if total_distance > 0:
            for length in segment_lengths:
                segment_time = (length / total_distance) * execution_time * 0.95  # Leave 5% margin
                segment_times.append(segment_time)
        else:
            # If total distance is 0 but there are multiple segments, split time evenly
            if num_segments > 0:
                time_per_segment = (execution_time * 0.95) / num_segments
                segment_times = [time_per_segment] * num_segments
            else:
                segment_times = []
        
        logger.debug(f"Search pattern: total_distance={total_distance:.1f}m, "
                    f"required_avg_speed={total_distance/execution_time:.1f}m/s")
        
        # Start motion simulation
        current_pos = np.array(start_pos)
        elapsed_time = 0.0
        path_index = 0
        
        # Search-specific phase information
        search_phases = ["scanning", "searching", "identifying"]
        current_phase_index = 0
        
        while elapsed_time < execution_time and path_index < len(path) - 1:
            # Check interrupt
            if interrupt_check():
                return {
                    "success": False,
                    "final_position": current_pos.tolist(),
                    "reason": "interrupted",
                    "actual_time": elapsed_time
                }
            
            # Current segment start and end
            segment_start = path[path_index]
            segment_end = path[path_index + 1]
            segment_time = segment_times[path_index]
            
            # Interpolate within current segment
            segment_elapsed = 0.0
            
            while segment_elapsed < segment_time and elapsed_time < execution_time:
                # Calculate current segment progress
                segment_progress = segment_elapsed / segment_time if segment_time > 0 else 1.0
                
                # Use smooth interpolation
                smooth_progress = smooth_step(segment_progress)
                current_pos = segment_start + (segment_end - segment_start) * smooth_progress
                
                # Calculate instantaneous velocity
                if segment_time > 0:
                    instant_velocity = np.linalg.norm(segment_end - segment_start) / segment_time
                else:
                    instant_velocity = 0.0
                
                # Calculate overall progress
                distance_traveled = sum(segment_lengths[:path_index]) + segment_lengths[path_index] * segment_progress
                overall_progress = distance_traveled / total_distance if total_distance > 0 else 1.0
                
                # Orientation
                direction = segment_end - segment_start
                if np.linalg.norm(direction) > 0.01:
                    orientation = math.atan2(direction[1], direction[0])
                else:
                    orientation = 0.0
                
                # Update search phase based on progress
                phase_progress = overall_progress * len(search_phases)
                current_phase_index = min(int(phase_progress), len(search_phases) - 1)
                current_phase = search_phases[current_phase_index]
                
                # Callback update position, using data format consistent with other skills
                update_callback({
                    "position": current_pos.tolist(),
                    "velocity": instant_velocity,
                    "orientation": orientation,
                    "progress": min(overall_progress, elapsed_time / execution_time),
                    "phase": current_phase,  # Search-specific phase information
                    "path_index": path_index,
                    "total_points": len(path),
                    "time_elapsed": elapsed_time,
                    "time_total": execution_time,
                    "coverage": overall_progress * 100  # Search coverage percentage
                })
                
                # Wait for next update cycle
                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                segment_elapsed += self.update_interval
                elapsed_time += self.update_interval
            
            # Move to next segment
            path_index += 1
        
        # Ensure final arrival at the last waypoint
        final_position = search_waypoints[-1] if search_waypoints else start_pos
        
        # Final update to ensure position and progress are both 100%
        update_callback({
            "position": final_position,
            "velocity": 0.0,
            "orientation": 0.0,
            "progress": 1.0,
            "phase": "completed",
            "path_index": len(path) - 1,
            "total_points": len(path),
            "time_elapsed": elapsed_time,
            "time_total": execution_time,
            "coverage": 100
        })
        
        logger.info(f"Search simulation completed. Final position: {final_position}, "
                f"actual_time: {elapsed_time:.1f}s")
        
        return {
            "success": True,
            "final_position": final_position,
            "actual_time": elapsed_time,
            "path_length": total_distance,
            "avg_velocity": total_distance / elapsed_time if elapsed_time > 0 else 0,
            "area_covered": True  # Search-specific return value
        }
    
    async def simulate_follow(
        self,
        robot_type: str,
        robot_pos: List[float],
        target_id: int,
        get_target_position: Callable[[], Optional[List[float]]],
        following_distance: float,
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Timed following: 30% fast approach + 70% distance maintenance.
        - End time is exactly execution_time.
        - If the target moves, each tick approaches the dynamic ideal follow position using remaining-time interpolation.
        """

        # Time split
        approach_ratio = 0.30
        maintain_ratio = 0.70
        approach_time = execution_time * approach_ratio
        maintain_time = execution_time * maintain_ratio

        elapsed_time = 0.0
        current_pos = np.array(robot_pos, dtype=float)
        prev_pos = current_pos.copy()

        # Record the latest direction for fallback when target and robot nearly overlap
        last_dir = np.array([1.0, 0.0], dtype=float)

        def _desired_follow_pos(tp: np.ndarray, anchor: np.ndarray) -> np.ndarray:
            """Compute ideal follow position: target position - unit direction * following distance.
            Direction primarily uses target - current; otherwise uses last_dir.
            """
            vec = tp - anchor
            n = np.linalg.norm(vec)
            direction = (vec / n) if n > 1e-6 else last_dir
            return tp - direction * following_distance

        # =============== Phase 1: Approach (30%) ===============
        phase_elapsed = 0.0
        while phase_elapsed < approach_time and elapsed_time < execution_time:
            if interrupt_check():
                return {
                    "success": False,
                    "reason": "interrupted",
                    "final_position": current_pos.tolist(),
                    "actual_time": elapsed_time
                }

            tp = get_target_position()
            if not tp:
                return {
                    "success": False,
                    "reason": "target_lost",
                    "final_position": current_pos.tolist(),
                    "actual_time": elapsed_time
                }
            tp = np.array(tp, dtype=float)

            # Update last_dir
            to_target = tp - current_pos
            norm_tt = np.linalg.norm(to_target)
            if norm_tt > 1e-6:
                last_dir = to_target / norm_tt

            # Dynamic ideal position, recomputed each tick so moving targets can be caught
            desired = _desired_follow_pos(tp, current_pos)

            # Key point: spread displacement over remaining time to reach the ideal position by phase end
            remaining = max(approach_time - phase_elapsed, 1e-6)
            # Fast approach: add some gain (>1), but still avoid overshoot through remaining-time clipping
            gain = 1.5
            step_vec = (desired - current_pos) * min(1.0, (self.update_interval * gain) / remaining)

            prev_pos = current_pos.copy()
            current_pos = current_pos + step_vec

            # Speed and orientation
            move_vec = current_pos - prev_pos
            inst_speed = float(np.linalg.norm(move_vec) / max(self.update_interval, 1e-6))
            orientation = math.atan2(move_vec[1], move_vec[0]) if inst_speed > 1e-6 else 0.0

            progress = min(1.0, elapsed_time / execution_time)
            phase_progress = min(1.0, phase_elapsed / approach_time) if approach_time > 0 else 1.0

            update_callback({
                "position": current_pos.tolist(),
                "velocity": inst_speed,
                "orientation": orientation,
                "target_distance": float(np.linalg.norm(tp - current_pos)),
                "following_distance_desired": following_distance,
                "following_status": "approaching",
                "phase": "approach",
                "progress": progress,
                "phase_progress": phase_progress,
                "time_elapsed": elapsed_time,
                "time_total": execution_time
            })

            if self.fast_mode:
                await asyncio.sleep(0)  # Almost no delay, but yields control
            else:
                await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
            elapsed_time += self.update_interval
            phase_elapsed += self.update_interval

        # =============== Phase 2: Maintain (70%) ===============
        phase_elapsed = 0.0
        while phase_elapsed < maintain_time and elapsed_time < execution_time:
            if interrupt_check():
                return {
                    "success": False,
                    "reason": "interrupted",
                    "final_position": current_pos.tolist(),
                    "actual_time": elapsed_time
                }

            tp = get_target_position()
            if not tp:
                return {
                    "success": False,
                    "reason": "target_lost",
                    "final_position": current_pos.tolist(),
                    "actual_time": elapsed_time
                }
            tp = np.array(tp, dtype=float)

            to_target = tp - current_pos
            norm_tt = np.linalg.norm(to_target)
            if norm_tt > 1e-6:
                last_dir = to_target / norm_tt

            desired = _desired_follow_pos(tp, current_pos)

            # Maintain phase also approaches by remaining time to stay near the ideal distance by phase end
            remaining = max(maintain_time - phase_elapsed, 1e-6)
            # Use time-weighted interpolation for stable convergence; step size increases as remaining time shrinks
            alpha = self.update_interval / (remaining + self.update_interval)
            step_vec = (desired - current_pos) * alpha

            prev_pos = current_pos.copy()
            current_pos = current_pos + step_vec

            move_vec = current_pos - prev_pos
            inst_speed = float(np.linalg.norm(move_vec) / max(self.update_interval, 1e-6))
            orientation = math.atan2(move_vec[1], move_vec[0]) if inst_speed > 1e-6 else 0.0

            # Use tolerance around target distance to mark "locked"
            dist_now = float(np.linalg.norm(tp - current_pos))
            tol = 0.2  # 20% tolerance, adjustable if needed
            locked = following_distance * (1 - tol) <= dist_now <= following_distance * (1 + tol)

            progress = min(1.0, elapsed_time / execution_time)
            phase_progress = min(1.0, phase_elapsed / maintain_time) if maintain_time > 0 else 1.0

            update_callback({
                "position": current_pos.tolist(),
                "velocity": inst_speed,
                "orientation": orientation,
                "target_distance": dist_now,
                "following_distance_desired": following_distance,
                "following_status": "locked" if locked else "adjusting",
                "phase": "maintain",
                "progress": progress,
                "phase_progress": phase_progress,
                "time_elapsed": elapsed_time,
                "time_total": execution_time
            })

            if self.fast_mode:
                await asyncio.sleep(0)  # Almost no delay, but yields control
            else:
                await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
            elapsed_time += self.update_interval
            phase_elapsed += self.update_interval

        # End-state evaluation
        final_tp = get_target_position()
        if final_tp:
            final_tp = np.array(final_tp, dtype=float)
            final_dist = float(np.linalg.norm(final_tp - current_pos))
            following_maintained = following_distance * 0.8 <= final_dist <= following_distance * 1.2
        else:
            final_dist = -1.0
            following_maintained = False

        return {
            "success": True,
            "final_position": current_pos.tolist(),
            "actual_time": elapsed_time,
            "final_distance": final_dist,
            "following_maintained": following_maintained
        }
    
    async def simulate_broadcast(
        self,
        robot_type: str,
        robot_pos: List[float],
        message: str,
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Simulate the broadcast skill.
        Args:
            robot_type: Robot type.
            robot_pos: Current robot position [x, y].
            message: Text to broadcast; passed back in callback for UI display.
            update_callback: Progress update callback.
            interrupt_check: Interrupt check function.
            execution_time: Total execution duration in seconds.
        Returns:
            Execution result dictionary.
        """

        phases = [
            {"name": "prepare", "duration_ratio": 0.1},   # Turn on amplifier / self-check / pre-alert tone
            {"name": "broadcast", "duration_ratio": 0.8}, # Formal broadcast with S-curve volume ramp
            {"name": "confirm", "duration_ratio": 0.1}    # Log record / ending tone
        ]

        elapsed_time = 0.0

        for phase in phases:
            phase_name = phase["name"]
            phase_duration = execution_time * phase["duration_ratio"]
            phase_elapsed = 0.0

            while phase_elapsed < phase_duration and elapsed_time < execution_time:
                if interrupt_check():
                    return {
                        "success": False,
                        "reason": "interrupted",
                        "final_position": robot_pos,
                        "actual_time": elapsed_time
                    }

                # Overall progress and phase progress
                progress_total = elapsed_time / execution_time if execution_time > 0 else 1.0
                progress_phase = phase_elapsed / phase_duration if phase_duration > 0 else 1.0

                if phase_name == "prepare":
                    update_callback({
                        "phase": "prepare",
                        "position": robot_pos,
                        "message": message,
                        "progress": progress_total,
                        "phase_progress": progress_phase,
                        "time_elapsed": elapsed_time,
                        "time_total": execution_time
                    })

                elif phase_name == "broadcast":
                    update_callback({
                        "phase": "broadcast",
                        "position": robot_pos,
                        "message": message,
                        "progress": progress_total,
                        "phase_progress": progress_phase,
                        "time_elapsed": elapsed_time,
                        "time_total": execution_time
                    })

                else:  # confirm
                    update_callback({
                        "phase": "confirm",
                        "position": robot_pos,
                        "message": message,
                        "progress": progress_total,
                        "phase_progress": progress_phase,
                        "time_elapsed": elapsed_time,
                        "time_total": execution_time
                    })

                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                phase_elapsed += self.update_interval
                elapsed_time += self.update_interval

        # Final frame: ensure convergence and 100% progress
        update_callback({
            "phase": "completed",
            "position": robot_pos,
            "message": message,
            "progress": 1.0,
            "phase_progress": 1.0,
            "time_elapsed": elapsed_time,
            "time_total": execution_time
        })

        logger.info("Broadcast completed.")

        return {
            "success": True,
            "actual_time": elapsed_time,
            "message": message
        }
    
    async def simulate_place(
        self,
        robot_type: str,
        robot_pos: List[float],
        object_start_pos: List[float],
        surface_pos: List[float],
        object_weight: float,
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """Simulate placing an object onto a surface."""
        phases = [
            {"name": "position", "duration_ratio": 0.4},
            {"name": "lower", "duration_ratio": 0.4},
            {"name": "release", "duration_ratio": 0.2}
        ]
        elapsed_time = 0.0
        
        for phase in phases:
            phase_duration = execution_time * phase["duration_ratio"]
            phase_elapsed = 0.0
            
            while phase_elapsed < phase_duration:
                if interrupt_check():
                    return {"success": False, "phase": phase["name"], "reason": "interrupted"}
                
                progress = phase_elapsed / phase_duration
                
                if phase["name"] == "position":
                    # Move object from start pos to above surface pos
                    current_obj_pos = interpolate_position(object_start_pos, surface_pos, progress)
                    update_callback({
                        "phase": "positioning", "progress": progress, "object_position": current_obj_pos
                    })
                elif phase["name"] == "lower":
                    # Lower the object onto the surface
                    current_obj_pos = list(surface_pos)
                    update_callback({
                        "phase": "lowering", "progress": progress, "object_position": current_obj_pos
                    })
                elif phase["name"] == "release":
                    # Release manipulator
                    update_callback({"phase": "releasing", "progress": progress})

                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                phase_elapsed += self.update_interval
                elapsed_time += self.update_interval

        return {"success": True, "actual_time": elapsed_time}

    async def simulate_handle_hazard(
        self,
        robot_type: str,
        robot_pos: List[float],
        hazard_pos: List[float],
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """Simulate handling a hazard."""
        phases = [
            {"name": "deploy", "duration_ratio": 0.2},
            {"name": "apply", "duration_ratio": 0.6},
            {"name": "retract", "duration_ratio": 0.2}
        ]
        elapsed_time = 0.0

        for phase in phases:
            phase_duration = execution_time * phase["duration_ratio"]
            phase_elapsed = 0.0
            
            while phase_elapsed < phase_duration:
                if interrupt_check():
                    return {"success": False, "phase": phase["name"], "reason": "interrupted"}
                
                progress = phase_elapsed / phase_duration
                
                if phase["name"] == "apply":
                    # Visualize an expanding effect
                    effect_radius = 10.0 * progress
                    update_callback({
                        "phase": "applying_treatment", "progress": progress, "effect_radius": effect_radius
                    })
                else:
                    update_callback({"phase": phase["name"], "progress": progress})

                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                phase_elapsed += self.update_interval
                elapsed_time += self.update_interval

        return {"success": True, "actual_time": elapsed_time}

    async def simulate_guide(
        self,
        robot_type: str,
        start_pos: List[float],
        end_pos: List[float],
        obstacles: List,
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Simulate guiding. The core motion is navigation.
        This method reuses the simulate_navigate logic for efficiency.
        """
        path = self._plan_path(start_pos, end_pos, obstacles, self.motion_profiles.get(robot_type))
        if not path:
             path = [np.array(start_pos), np.array(end_pos)]
        
        # Add direction to callback for the follower
        def enriched_update_callback(motion_data: Dict):
            current_pos = np.array(motion_data.get("position", start_pos))
            path_index = motion_data.get("path_index", 0)
            
            if path_index < len(path) - 1:
                 direction = path[path_index + 1] - path[path_index]
                 norm = np.linalg.norm(direction)
                 if norm > 0:
                     motion_data["direction"] = direction / norm
            update_callback(motion_data)

        return await self.simulate_navigate(
            robot_type=robot_type,
            start_pos=start_pos,
            end_pos=end_pos,
            obstacles=obstacles,
            update_callback=enriched_update_callback,
            interrupt_check=interrupt_check,
            execution_time=execution_time
        )
        
    # === Helper Methods ===
    
    def _plan_path(
        self,
        start: List[float],
        end: List[float],
        obstacles: List[ObstacleInfo],
        profile: Dict
    ) -> List[np.ndarray]:
        """Plan path."""
        start_array = np.array(start)
        end_array = np.array(end)

        if self.fast_mode:
            num_points = 10  # Adjust the number of interpolation points as needed
            path_array = np.linspace(start_array, end_array, num=num_points)
        
            # Convert np.array([[x0,y0],...]) back to list [array([x0,y0]), ...]
            return [path_array[i] for i in range(num_points)]
        else:
            # Standard mode: plan path by motion type
            motion_type = profile["motion_type"]

            # Normalize obstacles, converting circles to rectangles
            normalized_obstacles = self._normalize_obstacles(obstacles)

            if motion_type == MotionType.LINEAR:
                # Straight path
                return [start_array, end_array]

            elif motion_type == MotionType.BEZIER:
                # Bezier curve path, suitable for drones
                mid_point = (start_array + end_array) / 2
                mid_point[1] += 20  # Raise midpoint
                return generate_bezier_path(start, mid_point.tolist(), end, 20)

            elif motion_type == MotionType.DYNAMIC_AVOID:
                # Dynamic obstacle-avoidance path, suitable for ground vehicles, using normalized obstacles
                return self._generate_avoiding_path(start, end, normalized_obstacles, 20)

            return [start_array, end_array]

    def _generate_bezier_path(
        self,
        p0: List[float],
        p1: List[float],
        p2: List[float],
        num_points: int
    ) -> List[np.ndarray]:
        """Generate a quadratic Bezier curve path."""
        path = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = ((1 - t) ** 2) * np.array(p0) + \
                    2 * (1 - t) * t * np.array(p1) + \
                    (t ** 2) * np.array(p2)
            path.append(point)
        return path

    def _generate_avoiding_path(
        self,
        start: List[float],
        end: List[float],
        obstacles: List[ObstacleInfo],
        num_points: int
    ) -> List[np.ndarray]:
        """Generate an obstacle-avoidance path using a visibility graph and Dijkstra shortest path."""
        start_point = np.array(start)
        end_point = np.array(end)
        
        # Check whether the straight path is blocked
        if not self._path_intersects_obstacles(start_point, end_point, obstacles):
            return generate_straight_path(start_point, end_point, num_points)
        
        # Build visibility graph and find shortest path
        waypoints = self._visibility_graph_shortest_path(start_point, end_point, obstacles)
        
        if not waypoints:
            # If no path is found, return straight path to make debugging easier
            logger.error("No path found! Returning direct path")
            return generate_straight_path(start_point, end_point, num_points)
        
        # Smooth path with spline curve
        smooth_path = generate_smooth_path(waypoints, num_points)
        
        return smooth_path

    def _visibility_graph_shortest_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[ObstacleInfo]
    ) -> List[np.ndarray]:
        """Build a visibility graph and use Dijkstra's algorithm to find the shortest path."""
        
        safety_margin = 15.0
        
        logger.debug(f"Building visibility graph from {start} to {goal}")
        logger.debug(f"Number of obstacles: {len(obstacles)}")
        
        # First check whether the start and goal are inside obstacles.
        start_in_obstacle = point_in_any_obstacle(start, obstacles, margin=0)
        goal_in_obstacle = point_in_any_obstacle(goal, obstacles, margin=0)
        
        if start_in_obstacle:
            # logger.warning("Start point is inside an obstacle!")
            # Find the nearest safe point as the new start.
            start = self._find_nearest_safe_point(start, obstacles, safety_margin)
            logger.debug(f"Adjusted start point to: {start}")
        
        if goal_in_obstacle:
            # logger.warning("Goal point is inside an obstacle!")
            # Find the nearest safe point as the new goal.
            goal = self._find_nearest_safe_point(goal, obstacles, safety_margin)
            logger.debug(f"Adjusted goal point to: {goal}")
        
        # Step 1: build all graph nodes.
        nodes = []
        nodes.append(('start', start))
        nodes.append(('goal', goal))
        
        # Add obstacle vertices with multiple margins to improve connectivity.
        for i, obs in enumerate(obstacles):
            # All obstacles now have min_corner and max_corner.
            min_corner = np.array(obs.shape["min_corner"])
            max_corner = np.array(obs.shape["max_corner"])
            
            # Use multiple margin levels.
            for margin_factor in [1.0, 1.5]:  # 15 m and 22.5 m
                current_margin = safety_margin * margin_factor
                expanded_min = min_corner - current_margin
                expanded_max = max_corner + current_margin
                
                # 8 key points.
                corners = [
                    ('obs_{}_bl_{}'.format(i, margin_factor), np.array([expanded_min[0], expanded_min[1]])),
                    ('obs_{}_br_{}'.format(i, margin_factor), np.array([expanded_max[0], expanded_min[1]])),
                    ('obs_{}_tr_{}'.format(i, margin_factor), np.array([expanded_max[0], expanded_max[1]])),
                    ('obs_{}_tl_{}'.format(i, margin_factor), np.array([expanded_min[0], expanded_max[1]])),
                ]
                
                # Add only points that are not inside other obstacles.
                for name, point in corners:
                    if not point_in_any_obstacle(point, obstacles, margin=5.0):
                        nodes.append((name, point))
        
        # Add extra path points to improve connectivity.
        # Add intermediate points between the start and goal.
        num_intermediate = 5
        for i in range(1, num_intermediate):
            t = i / num_intermediate
            intermediate = start * (1 - t) + goal * t
            
            # Add the intermediate point if it is not inside an obstacle.
            if not point_in_any_obstacle(intermediate, obstacles, margin=safety_margin):
                nodes.append(('inter_{}'.format(i), intermediate))
            else:
                # Try offsetting in the perpendicular direction.
                direction = goal - start
                perp = np.array([-direction[1], direction[0]])
                perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 0 else np.array([1, 0])
                
                for offset in [safety_margin * 2, -safety_margin * 2, safety_margin * 3, -safety_margin * 3]:
                    offset_point = intermediate + perp * offset
                    if not point_in_any_obstacle(offset_point, obstacles, margin=safety_margin):
                        nodes.append(('inter_{}_{}'.format(i, offset), offset_point))
                        break
        
        logger.debug(f"Total nodes in graph: {len(nodes)}")
        
        # Step 2: build edges with a looser connection strategy.
        edges = []
        n = len(nodes)
        edge_count = 0
        
        # Find at least k nearest reachable neighbors for each node.
        k_nearest = min(10, n - 1)  # Try connecting the 10 nearest nodes.
        
        for i in range(n):
            node_i_pos = nodes[i][1]
            
            # Calculate distances to all other nodes.
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(nodes[j][1] - node_i_pos)
                    distances.append((dist, j))
            
            # Sort and try connecting the nearest k nodes.
            distances.sort()
            connected_count = 0
            
            for dist, j in distances:
                if connected_count >= k_nearest:
                    break
                    
                node_j_pos = nodes[j][1]
                
                # Use slightly looser collision checks.
                if self._is_edge_valid_relaxed(node_i_pos, node_j_pos, obstacles, safety_margin * 0.8):
                    # Avoid duplicate edges.
                    if not any((e[0] == i and e[1] == j) or (e[0] == j and e[1] == i) for e in edges):
                        edges.append((i, j, dist))
                        edge_count += 1
                        connected_count += 1
        
        logger.debug(f"Total edges in graph: {edge_count}")
        
        # Check connectivity.
        start_edges = sum(1 for e in edges if e[0] == 0 or e[1] == 0)
        goal_edges = sum(1 for e in edges if e[0] == 1 or e[1] == 1)
        
        logger.debug(f"Start node has {start_edges} connections")
        logger.debug(f"Goal node has {goal_edges} connections")
        
        if start_edges == 0:
            # Force-connect the start to the nearest reachable node.
            nearest_idx, nearest_dist = self._find_nearest_reachable_node(
                start, nodes[1:], obstacles, safety_margin * 0.5
            )
            if nearest_idx is not None:
                edges.append((0, nearest_idx + 1, nearest_dist))
                logger.debug(f"Force-connected start to node {nearest_idx + 1}")
            else:
                logger.error("Cannot connect start point to any node!")
                return self._emergency_path(start, goal, obstacles)
        
        if goal_edges == 0:
            # Force-connect the goal.
            nearest_idx, nearest_dist = self._find_nearest_reachable_node(
                goal, nodes[:-1], obstacles, safety_margin * 0.5
            )
            if nearest_idx is not None:
                edges.append((1, nearest_idx, nearest_dist))
                logger.debug(f"Force-connected goal to node {nearest_idx}")
        
        # Use Dijkstra's algorithm.
        path_indices = dijkstra(n, edges, 0, 1)
        
        if not path_indices:
            logger.error("Dijkstra failed to find path, using emergency path")
            return self._emergency_path(start, goal, obstacles)
        
        # Convert to coordinate points.
        path = [nodes[i][1] for i in path_indices]
        return path

    def _find_nearest_safe_point(
        self,
        point: np.ndarray,
        obstacles: List[ObstacleInfo],
        margin: float
    ) -> np.ndarray:
        """Find the nearest safe point that is not inside an obstacle."""
        
        # Try multiple directions.
        directions = [
            np.array([1, 0]), np.array([-1, 0]),
            np.array([0, 1]), np.array([0, -1]),
            np.array([1, 1]), np.array([1, -1]),
            np.array([-1, 1]), np.array([-1, -1])
        ]
        
        # Increase the distance step by step.
        for distance in range(10, 200, 10):
            for direction in directions:
                test_point = point + direction * distance
                if not point_in_any_obstacle(test_point, obstacles, margin=margin):
                    return test_point
        
        # If none is found, return the original point to avoid getting stuck.
        return point

    def _find_nearest_reachable_node(
        self,
        from_point: np.ndarray,
        nodes: List[Tuple[str, np.ndarray]],
        obstacles: List[ObstacleInfo],
        margin: float
    ) -> Tuple[Optional[int], float]:
        """Find the nearest reachable node."""
        
        best_idx = None
        best_dist = float('inf')
        
        for i, (name, node_pos) in enumerate(nodes):
            dist = np.linalg.norm(node_pos - from_point)
            
            # Check reachability with a looser margin.
            if self._is_edge_valid_relaxed(from_point, node_pos, obstacles, margin):
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        
        return best_idx, best_dist

    def _is_edge_valid_relaxed(
        self,
        start: np.ndarray,
        end: np.ndarray,
        obstacles: List[ObstacleInfo],
        safety_margin: float
    ) -> bool:
        """Looser edge validation used to ensure connectivity."""
        
        # Only check whether the segment intersects obstacles with a smaller safety margin.
        for obs in obstacles:
            expanded_obs = ObstacleInfo(
                position=obs.position,
                shape={
                    "min_corner": (np.array(obs.shape["min_corner"]) - safety_margin).tolist(),
                    "max_corner": (np.array(obs.shape["max_corner"]) + safety_margin).tolist()
                }
            )
            
            if line_intersects_obstacle(start.tolist(), end.tolist(), expanded_obs):
                return False
        
        return True

    def _emergency_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[ObstacleInfo]
    ) -> List[np.ndarray]:
        """Emergency path used when all other methods fail."""
        
        logger.warning("Using emergency path generation")
        
        # Simple grid search.
        path = [start]
        current = start.copy()
        
        # Step size.
        step_size = 20.0
        max_steps = 100
        
        for _ in range(max_steps):
            # Calculate the direction to the goal.
            direction = goal - current
            distance = np.linalg.norm(direction)
            
            if distance < step_size:
                path.append(goal)
                break
            
            direction = direction / distance * step_size
            
            # Try moving directly toward the goal.
            next_point = current + direction
            
            if not point_in_any_obstacle(next_point, obstacles, margin=10.0):
                path.append(next_point)
                current = next_point
            else:
                # Try a detour.
                found = False
                for angle in [90, -90, 45, -45, 135, -135]:
                    rad = np.radians(angle)
                    rotated_dir = np.array([
                        direction[0] * np.cos(rad) - direction[1] * np.sin(rad),
                        direction[0] * np.sin(rad) + direction[1] * np.cos(rad)
                    ])
                    
                    test_point = current + rotated_dir
                    if not point_in_any_obstacle(test_point, obstacles, margin=10.0):
                        path.append(test_point)
                        current = test_point
                        found = True
                        break
                
                if not found:
                    # Random direction.
                    random_angle = np.random.uniform(0, 2 * np.pi)
                    random_dir = np.array([np.cos(random_angle), np.sin(random_angle)]) * step_size
                    next_point = current + random_dir
                    path.append(next_point)
                    current = next_point
        
        # Ensure the goal is included.
        if np.linalg.norm(path[-1] - goal) > 0.1:
            path.append(goal)
        
        return path

    def _generate_smooth_path(
        self,
        waypoints: List[np.ndarray],
        num_points: int
    ) -> List[np.ndarray]:
        """Generate a smooth path with cubic Bezier curves."""
        
        if len(waypoints) <= 2:
            return generate_straight_path(waypoints[0], waypoints[-1], num_points)
        
        # Generate a Bezier curve for each adjacent waypoint pair.
        smooth_path = []
        segments = len(waypoints) - 1
        points_per_segment = max(3, num_points // segments)
        
        for i in range(segments):
            p0 = waypoints[i]
            p3 = waypoints[i + 1]
            
            # Calculate control points.
            if i == 0:
                # First segment.
                direction = waypoints[i + 1] - waypoints[i]
                p1 = p0 + direction * 0.3
            else:
                # Use the previous point to calculate the tangent.
                tangent = (waypoints[i + 1] - waypoints[i - 1]) / 2
                p1 = p0 + tangent * 0.3
            
            if i == segments - 1:
                # Last segment.
                direction = waypoints[i + 1] - waypoints[i]
                p2 = p3 - direction * 0.3
            else:
                # Use the next point to calculate the tangent.
                tangent = (waypoints[i + 2] - waypoints[i]) / 2
                p2 = p3 - tangent * 0.3
            
            # Generate Bezier curve points.
            segment_points = points_per_segment
            if i == segments - 1:
                segment_points = num_points - len(smooth_path)
            
            for j in range(segment_points):
                t = j / (segment_points - 1) if segment_points > 1 else 0
                
                # Cubic Bezier curve formula.
                point = ((1 - t) ** 3) * p0 + \
                        3 * ((1 - t) ** 2) * t * p1 + \
                        3 * (1 - t) * (t ** 2) * p2 + \
                        (t ** 3) * p3
                
                # Avoid duplicate points.
                if not smooth_path or np.linalg.norm(point - smooth_path[-1]) > 0.1:
                    smooth_path.append(point)
        
        # Ensure the goal is exact.
        if len(smooth_path) > 0 and np.linalg.norm(smooth_path[-1] - waypoints[-1]) > 0.1:
            smooth_path.append(waypoints[-1])
        
        return smooth_path

    def _path_intersects_obstacles(
        self,
        start: np.ndarray,
        end: np.ndarray,
        obstacles: List[ObstacleInfo]
    ) -> bool:
        """Check whether the path intersects any obstacle."""
        for obs in obstacles:
            if line_intersects_obstacle(start.tolist(), end.tolist(), obs):
                return True
        return False
    
    def _normalize_obstacles(self, obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """Normalize all obstacles to rectangles, converting circles to bounding boxes."""
        normalized_obstacles = []
        
        for obs in obstacles:
            if obs.shape.get('type') == 'circle':
                # Convert a circular obstacle to a bounding rectangle.
                center = np.array(obs.shape['center'])
                radius = obs.shape['radius']
                
                # Create a new rectangular obstacle.
                new_shape = {
                    'type': 'rectangle',
                    'min_corner': [center[0] - radius, center[1] - radius],
                    'max_corner': [center[0] + radius, center[1] + radius],
                    'original_type': 'circle',  # Keep original type information.
                    'original_center': obs.shape['center'],
                    'original_radius': radius
                }
                
                # Create a new ObstacleInfo object.
                normalized_obs = ObstacleInfo(
                    position=obs.position,
                    shape=new_shape,
                    is_dynamic=obs.is_dynamic,
                    velocity=obs.velocity
                )
                normalized_obstacles.append(normalized_obs)
            else:
                # Add other types, such as rectangles, directly.
                normalized_obstacles.append(obs)
        
        return normalized_obstacles
    
    def _update_motion_state(
        self,
        current: MotionState,
        target: np.ndarray,
        profile: Dict,
        dt: float
    ) -> MotionState:
        """
        Update motion state with a physics-based motion model.
        
        Note: the navigate skill no longer uses this method. It uses
        time-based interpolation instead. This method is mainly for other
        scenarios that need physics simulation.
        """
        # Calculate the direction to the target.
        direction = target - current.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.01:
            return current
        
        direction_normalized = direction / distance
        
        # === Improved velocity calculation ===
        # Adjust desired velocity dynamically based on remaining distance.
        max_velocity = profile["max_velocity"]
        max_acceleration = profile["max_acceleration"]
        
        # Calculate braking distance.
        braking_distance = (np.linalg.norm(current.velocity) ** 2) / (2 * max_acceleration)
        
        # Start decelerating when close to the target.
        if distance < braking_distance * 1.5:
            # Use smooth deceleration.
            desired_speed = max_velocity * (distance / (braking_distance * 1.5))
            desired_speed = max(desired_speed, 1.0)  # Minimum speed 1 m/s.
        else:
            # Normal speed.
            desired_speed = max_velocity
        
        # Calculate desired velocity vector.
        desired_velocity = direction_normalized * desired_speed
        
        # Apply acceleration limits.
        velocity_change = desired_velocity - current.velocity
        max_velocity_change = max_acceleration * dt
        
        if np.linalg.norm(velocity_change) > max_velocity_change:
            velocity_change = velocity_change / np.linalg.norm(velocity_change) * max_velocity_change
        
        new_velocity = current.velocity + velocity_change
        
        # Limit max speed.
        speed = np.linalg.norm(new_velocity)
        if speed > max_velocity:
            new_velocity = new_velocity / speed * max_velocity
        
        # Update position.
        new_position = current.position + new_velocity * dt
        
        # Ensure the target is not overshot.
        if np.linalg.norm(new_position - current.position) > distance:
            new_position = target
            new_velocity = np.zeros(2)
        
        # Update orientation.
        if np.linalg.norm(new_velocity) > 0.01:
            new_orientation = math.atan2(new_velocity[1], new_velocity[0])
        else:
            new_orientation = current.orientation
        
        return MotionState(
            position=new_position,
            velocity=new_velocity,
            acceleration=velocity_change / dt,
            orientation=new_orientation,
            timestamp=current.timestamp + dt
        )
