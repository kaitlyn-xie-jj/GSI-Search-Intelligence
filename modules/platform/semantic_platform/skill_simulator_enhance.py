import asyncio
from numpy._typing._array_like import NDArray
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import math
from .utils.motion_math import (
    calculate_path_length
)
from modules.platform.semantic_platform.skill_simulator import SkillMotionSimulator, MotionType, ObstacleInfo

logger = logging.getLogger(__name__)

# ============= Entity Motion Type Enum =============
class EntityMotionType(Enum):
    """Entity motion type."""
    SKILL_EXECUTION = "skill_execution"  # Motion caused by skill execution
    AUTONOMOUS = "autonomous"  # Autonomous motion, such as target vehicle movement
    DISABLED = "disabled"  # Disabled state, cannot move
    EXTERNAL_CONTROL = "external_control"  # External control motion

# ============= Entity State Manager =============
class EntityStateManager:
    """
    Manage states for all entities, including robots and other dynamic entities.
    """
    
    def __init__(self):
        """Initialize the entity state manager."""
        self.entity_states = {}  # entity_id -> EntityState
        self.disabled_entities = set()  # Disabled entity set
        self.entity_properties = {}  # entity_id -> properties dict
        
    def set_entity_disabled(self, entity_id: int, disabled: bool = True):
        """
        Set entity disabled state.
        
        Args:
            entity_id: Entity ID.
            disabled: Whether the entity is disabled.
        """
        if disabled:
            self.disabled_entities.add(entity_id)
            logger.warning(f"Entity {entity_id} has been disabled")
        else:
            self.disabled_entities.discard(entity_id)
            logger.info(f"Entity {entity_id} has been re-enabled")
    
    def is_entity_disabled(self, entity_id: int) -> bool:
        """
        Check whether the entity is disabled.
        
        Args:
            entity_id: Entity ID.
            
        Returns:
            bool: Whether the entity is disabled.
        """
        return entity_id in self.disabled_entities
    
    def can_entity_move(self, entity_id: int) -> bool:
        """
        Check whether the entity can move.
        
        Args:
            entity_id: Entity ID.
            
        Returns:
            bool: Whether the entity can move.
        """
        return entity_id not in self.disabled_entities
    
    def set_entity_property(self, entity_id: int, property_name: str, value: Any):
        """
        Set an entity property.
        
        Args:
            entity_id: Entity ID.
            property_name: Property name.
            value: Property value.
        """
        if entity_id not in self.entity_properties:
            self.entity_properties[entity_id] = {}
        self.entity_properties[entity_id][property_name] = value
    
    def get_entity_property(self, entity_id: int, property_name: str, default=None):
        """
        Get an entity property.
        
        Args:
            entity_id: Entity ID.
            property_name: Property name.
            default: Default value.
            
        Returns:
            Property value.
        """
        if entity_id in self.entity_properties:
            return self.entity_properties[entity_id].get(property_name, default)
        return default
    
    def clear_entity(self, entity_id: int):
        """
        Clear all state information for an entity.
        
        Args:
            entity_id: Entity ID.
        """
        self.disabled_entities.discard(entity_id)
        self.entity_states.pop(entity_id, None)
        self.entity_properties.pop(entity_id, None)

# ============= Enhanced Motion Simulator =============
class EnhancedSkillMotionSimulator(SkillMotionSimulator):
    """
    Enhanced motion simulator that supports entity motion under new cases.
    Inherits from SkillMotionSimulator.
    """
    
    def __init__(self, update_interval: float = 0.1, fast_mode: bool = False):
        """
        Initialize the enhanced motion simulator.
        
        Args:
            update_interval: Update interval in seconds.
            fast_mode: Fast mode; skip step-by-step simulation and compute only key frames.
        """
        super().__init__(update_interval, fast_mode)
        # Fast mode flag
        self.fast_mode = fast_mode
        
        # Entity state manager
        self.entity_state_manager = EntityStateManager()
        
        # Add motion profiles for non-robot entities
        self.entity_motion_profiles = {
            "VEHICLE": {  # Target vehicle
                "max_velocity": 25.0,  # m/s
                "max_acceleration": 5.0,  # m/s^2
                "max_turn_rate": math.pi / 6,  # rad/s
                "motion_type": MotionType.DYNAMIC_AVOID
            },
            "CARGO": {  # Cargo, usually passively moved
                "max_velocity": 0.0,
                "max_acceleration": 0.0,
                "motion_type": MotionType.LINEAR
            },
            "PERSON": {  # Person
                "max_velocity": 5.0,  # m/s, walking speed
                "max_acceleration": 2.0,  # m/s^2
                "max_turn_rate": math.pi / 3,
                "motion_type": MotionType.DYNAMIC_AVOID
            },
            "DRONE": {  # Drone, non-robot
                "max_velocity": 40.0,  # m/s
                "max_acceleration": 8.0,  # m/s^2
                "max_turn_rate": math.pi / 2,
                "motion_type": MotionType.BEZIER
            }
        }
        
        logger.info("EnhancedSkillMotionSimulator initialized")
    
    async def simulate_autonomous_movement(
        self,
        entity_id: int,
        entity_type: str,
        start_pos: List[float],
        end_pos: List[float],
        update_callback: Callable[[Dict], None],
        interrupt_check: Callable[[], bool],
        movement_time: float,
        reason: str = "autonomous",
        obstacles: Optional[List[ObstacleInfo]] = None
    ) -> Dict[str, Any]:
        """
        Simulate autonomous entity motion, such as target vehicle movement.
        
        Args:
            entity_id: Entity ID.
            entity_type: Entity type (VEHICLE, CARGO, PERSON, etc.).
            start_pos: Start position [x, y].
            end_pos: Target position [x, y].
            update_callback: Position update callback.
            interrupt_check: Interrupt check function.
            movement_time: Movement time in seconds.
            reason: Movement reason.
            obstacles: Obstacle list (optional).
            
        Returns:
            Dict: Execution result.
                - success: Whether execution succeeded.
                - final_position: Final position.
                - actual_time: Actual elapsed time.
                - reason: Failure reason if execution failed.
        """
        # Check whether the entity is disabled
        if self.entity_state_manager.is_entity_disabled(entity_id):
            logger.warning(f"Entity {entity_id} is disabled, cannot move")
            return {
                'success': False,
                'reason': 'entity_disabled',
                'final_position': start_pos
            }
        
        # Get motion profile
        profile = self.entity_motion_profiles.get(
            entity_type, 
            self.entity_motion_profiles["VEHICLE"]
        )
        
        # Calculate path
        distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        
        logger.info(f"Autonomous movement: entity {entity_id} ({entity_type}) "
                   f"moving from {start_pos} to {end_pos}, "
                   f"distance: {distance:.1f}m, time: {movement_time:.1f}s, "
                   f"reason: {reason}")
        
        # In fast_mode, use linear interpolation and skip complex path planning
        if self.fast_mode:
            # Use numpy linspace to generate 5 interpolated points: start + 3 middle points + end
            num_points = 5
            start_array = np.array(start_pos)
            end_array = np.array(end_pos)
            path_points = []
            for i in range(len(start_array)):
                path_points.append(np.linspace(start_array[i], end_array[i], num_points))
            path_array = np.array(path_points).T
            path: list[NDArray[Any]] = [path_array[i] for i in range(num_points)]
        else:
            # Standard mode: choose path planning strategy by entity type
            if obstacles and profile["motion_type"] == MotionType.DYNAMIC_AVOID:
                # Use obstacle-avoidance path
                path = self._plan_path(start_pos, end_pos, obstacles, profile)
            elif profile["motion_type"] == MotionType.BEZIER:
                # Use Bezier curve, suitable for aerial units
                mid_point = (np.array(start_pos) + np.array(end_pos)) / 2
                mid_point = mid_point.tolist()
                if entity_type == "DRONE":
                    mid_point[1] += 20  # Raise midpoint
                path = self._generate_bezier_path(start_pos, mid_point, end_pos, 20)
            else:
                # Straight path
                path = self._generate_straight_path(
                    np.array(start_pos), 
                    np.array(end_pos), 
                    20
                )
        
        # Ensure path is not empty
        if not path:
            path = [np.array(start_pos), np.array(end_pos)]
        
        # Calculate total path length
        path_length = calculate_path_length(path)
        
        # ===== Unified step-by-step simulation shared by standard and fast modes =====
        # fast_mode affects only sleep time, not calculation logic
        # Execute motion simulation
        current_pos = np.array(start_pos)
        elapsed_time = 0.0
        path_index = 0
        
        # Calculate time allocation for each segment
        segment_times = []
        if len(path) > 1:
            for i in range(len(path) - 1):
                segment_length = np.linalg.norm(path[i + 1] - path[i])
                segment_time = (segment_length / path_length) * movement_time if path_length > 0 else movement_time / (len(path) - 1)
                segment_times.append(segment_time)
        else:
            segment_times = [movement_time]
        
        while elapsed_time < movement_time and path_index < len(path) - 1:
            # Check interrupt
            if interrupt_check():
                logger.info(f"Entity {entity_id} movement interrupted")
                return {
                    'success': False,
                    'reason': 'interrupted',
                    'final_position': current_pos.tolist()
                }
            
            # Check again whether the entity became disabled during movement
            if self.entity_state_manager.is_entity_disabled(entity_id):
                logger.warning(f"Entity {entity_id} disabled during movement")
                return {
                    'success': False,
                    'reason': 'disabled_during_movement',
                    'final_position': current_pos.tolist()
                }
            
            # Current segment start and end
            segment_start = path[path_index]
            segment_end = path[path_index + 1]
            segment_time = segment_times[path_index]
            
            # Interpolate within current segment
            segment_elapsed = 0.0
            while segment_elapsed < segment_time and elapsed_time < movement_time:
                # Calculate current segment progress
                segment_progress = segment_elapsed / segment_time if segment_time > 0 else 1.0
                
                # Use smooth interpolation
                smooth_progress = self._smooth_step(segment_progress)
                current_pos = segment_start + (segment_end - segment_start) * smooth_progress
                
                # Calculate overall progress
                overall_progress = elapsed_time / movement_time if movement_time > 0 else 1.0
                
                # Calculate orientation
                direction = segment_end - segment_start
                if np.linalg.norm(direction) > 0.01:
                    orientation = math.atan2(direction[1], direction[0])
                else:
                    orientation = 0.0
                
                # Calculate instantaneous velocity
                if segment_time > 0:
                    instant_velocity = np.linalg.norm(segment_end - segment_start) / segment_time
                else:
                    instant_velocity = 0.0
                
                # Callback update
                update_callback({
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'position': current_pos.tolist(),
                    'velocity': instant_velocity,
                    'orientation': orientation,
                    'progress': overall_progress,
                    'motion_type': EntityMotionType.AUTONOMOUS.value,
                    'reason': reason,
                    'path_index': path_index,
                    'total_points': len(path)
                })
                
                if self.fast_mode:
                    await asyncio.sleep(0)  # Almost no delay, but yields control
                else:
                    await asyncio.sleep(self.update_interval)  # Standard mode: wait fixed time
                
                segment_elapsed += self.update_interval
                elapsed_time += self.update_interval
            
            # Move to next segment
            path_index += 1
        
        # Ensure arrival at the endpoint
        final_position = end_pos
        
        # Final update
        update_callback({
            'entity_id': entity_id,
            'entity_type': entity_type,
            'position': final_position,
            'velocity': 0.0,
            'orientation': math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0]),
            'progress': 1.0,
            'motion_type': EntityMotionType.AUTONOMOUS.value,
            'reason': reason
        })
        
        logger.info(f"Entity {entity_id} autonomous movement completed")
        
        return {
            'success': True,
            'final_position': final_position,
            'actual_time': elapsed_time,
            'path_length': path_length,
            'avg_velocity': path_length / elapsed_time if elapsed_time > 0 else 0
        }
    
    def set_entity_motion_profile(self, entity_type: str, profile: Dict[str, Any]):
        """
        Set or update the motion profile for an entity type.
        
        Args:
            entity_type: Entity type.
            profile: Motion profile dictionary.
        """
        self.entity_motion_profiles[entity_type] = profile
        logger.info(f"Motion profile updated for entity type: {entity_type}")
    
    def get_entity_motion_profile(self, entity_type: str) -> Dict[str, Any]:
        """
        Get the motion profile for an entity type.
        
        Args:
            entity_type: Entity type.
            
        Returns:
            Motion profile dictionary.
        """
        return self.entity_motion_profiles.get(
            entity_type, 
            self.entity_motion_profiles["VEHICLE"]
        )
