# -*- coding: utf-8 -*-
"""
Plan translation pipeline - convert allocation results into executable skill sequences.
"""
from typing import Dict, Any, Optional, List, Tuple
from .plan_translator import PlanTranslator
from .utils.multi_robot_coord_utils import coordinate_multi_robot_params
from modules.platform.abstract_scene_graph import AbstractSceneGraph


class PlanTranslateCoordinator:
    """
    Plan translation coordinator.
    """
    
    # Energy consumed per unit distance.
    ENERGY_COEFFICIENT = 1.0
    
    def __init__(self, scene_graph: AbstractSceneGraph, label_to_id_map: Dict, id_to_label_map: Dict):
        self.scene_graph = scene_graph
        self.label_to_id_map = label_to_id_map
        self.id_to_label_map = id_to_label_map
        self.translator = PlanTranslator(scene_graph, label_to_id_map, id_to_label_map)
        self._last_plan_energy = 0.0  # Energy consumption from the latest translation.

    def run(
        self,
        dispatcher_result: Dict[str, Any],
        skill_schemas: Dict[str, Any],
        dependencies: Optional[List[Tuple[str, str]]] = None, 
        category_map: Optional[Dict[str, str]] = None,
        goal_cfg: Optional[Dict[str, Any]] = None,
        area_boundaries: Optional[Dict[str, Any]] = None,
        runtime_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run plan translation.
        
        Args:
            dispatcher_result: Dispatcher result containing timestep_skills.
            skill_schemas: Skill schema definitions.
            dependencies: Task dependencies, kept for the interface and not used yet.
            category_map: Category mapping.
            goal_cfg: Goal configuration.
            area_boundaries: Area boundaries.
            runtime_params: Runtime parameters.
            
        Returns:
            Parsed skill dictionary organized by timestep.
            {
                0: {"UGV_01": {"skill": "navigate", "params": {...}}, ...},
                1: {"Humanoid_01": {"skill": "place", "params": {...}}},
                ...
            }
        """
        # Get the skill list organized by timestep.
        timestep_skills = dispatcher_result.get("timestep_skills", {})
        if not timestep_skills:
            return {}
        
        # 1) Translate skill parameters.
        translated_skills = self.translator.translate_timestep_skills(
            timestep_skills=timestep_skills,
            skill_schemas=skill_schemas,
            category_map=category_map or {},
            goal_cfg=goal_cfg or {},
            area_boundaries=area_boundaries or {},
            runtime_params=runtime_params or {}
        )
        
        # 2) Coordinate multi-robot parameters, such as search area splitting.
        skills_by_timestep: Dict[int, Dict[str, Any]] = {}
        for timestep_str, robot_skills in translated_skills.items():
            timestep = int(timestep_str)
            robot_skill_pairs = [
                (robot_label, skill_info) 
                for robot_label, skill_info in robot_skills.items()
            ]
            
            # Run multi-robot parameter coordination.
            coordinated_pairs = coordinate_multi_robot_params(robot_skill_pairs)
            
            # Rebuild the timestep dictionary.
            skills_by_timestep[timestep] = {}
            for robot_label, skill_info in coordinated_pairs:
                skills_by_timestep[timestep][robot_label] = {
                    "skill": skill_info["skill"],
                    "params": skill_info.get("params", {}) or {}
                }
        
        sorted_result = dict(sorted(skills_by_timestep.items()))
        
        # 3) Compute energy consumption for movement skills only: navigate and guide.
        self._last_plan_energy = self._compute_plan_energy(sorted_result)
        
        return sorted_result

    def get_last_plan_energy(self) -> float:
        """Get the energy consumption from the latest plan translation."""
        return self._last_plan_energy

    def _compute_plan_energy(self, skills_by_timestep: Dict[int, Dict[str, Any]]) -> float:
        """Compute total plan energy consumption.
        
        Only movement skills (navigate, guide) are counted:
        energy = Euclidean distance from robot position to destination * ENERGY_COEFFICIENT.
        
        Args:
            skills_by_timestep: Translated skill sequence.
            
        Returns:
            Total energy consumption.
        """
        from math import hypot
        from modules.utils.location_utils import get_entity_position
        
        movement_skills = {"navigate", "goto", "go_to", "guide"}
        total_energy = 0.0
        
        for _ts, robot_skills in skills_by_timestep.items():
            for robot_label, skill_info in robot_skills.items():
                skill_name = skill_info.get("skill", "").lower()
                if skill_name not in movement_skills:
                    continue
                
                params = skill_info.get("params", {})
                dest = params.get("dest")
                if not dest or "x" not in dest or "y" not in dest:
                    continue
                
                # Get the robot's current position.
                robot_id = self.label_to_id_map.get(robot_label)
                robot_pos = get_entity_position(self.scene_graph, robot_id)
                if not robot_pos:
                    continue
                
                dist = hypot(
                    float(dest["x"]) - float(robot_pos[0]),
                    float(dest["y"]) - float(robot_pos[1]),
                )
                total_energy += dist * self.ENERGY_COEFFICIENT
        
        return round(total_energy, 4)
