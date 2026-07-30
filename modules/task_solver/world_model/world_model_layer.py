import logging
import asyncio
from typing import Any, Dict, List, Optional

from modules.task_solver.world_model.plan_process_pipeline import PlanTranslateCoordinator
from modules.task_solver.world_model.goal_progress_monitor import GoalProgressMonitor
from modules.task_solver.world_model.world_model_manager import WorldModelManager
from modules.task_solver.sgi_planner.prompt import SKILL_SCHEMAS
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.utils.system.var_dump import dump_var
from modules.utils.system.logging_utils import dlog
from modules.config.system_config import config

logger = logging.getLogger(__name__)


class WorldModelLayer:
    """
    World model layer for plan understanding, state tracking, and goal monitoring.
    """
    
    def __init__(self, scene_graph: AbstractSceneGraph,
                 monitor, logger, is_replay: bool = False):
        """
        Initialize the world model layer.
        
        Args:
            scene_graph: Scene graph, using AbstractSceneGraph.
            monitor: Task monitor.
            logger: Logger.
            is_replay: Whether replay mode is enabled.
        """
        self.scene_graph: AbstractSceneGraph = scene_graph
        self.monitor = monitor
        self.logger = logger
        self.knowledge_scope = 'local'
        self.is_replay = is_replay
        
        # World model manager.
        self.world_model_manager = WorldModelManager(
            scene_graph=scene_graph,
            logger=logger
        )
        
        # Mapping tables.
        self.label_to_id_map = scene_graph.get_node_map(map_type='label_to_id') if scene_graph else {}
        self.id_to_label_map = scene_graph.get_node_map(map_type='id_to_label') if scene_graph else {}
        
        # Plan translation coordinator, including parameter processing.
        self.plan_translate_coordinator = PlanTranslateCoordinator(
            scene_graph=scene_graph,
            label_to_id_map=self.label_to_id_map,
            id_to_label_map=self.id_to_label_map
        )
        
        # Goal progress monitor.
        self.goal_progress_monitor = GoalProgressMonitor(scene_graph=scene_graph)

        # Link the monitor to the goal progress monitor.
        if self.monitor:
            self.monitor.set_goal_progress_monitor(self.goal_progress_monitor)
        
        # Register the current goal.
        goal = self.scene_graph.get_goal() if self.scene_graph else None
        if self.monitor and self.scene_graph and goal:
            self.goal_progress_monitor.register_goal(
                goal_config=goal
            )
            dump_var("goal", goal) # Record the current goal.
    
    def set_knowledge_scope(self, scope: str):
        """Set the knowledge scope."""
        if scope not in ['local', 'global']:
            logger.warning(f"Invalid knowledge scope: {scope}. Using 'local' as default.")
            scope = 'local'
        self.knowledge_scope = scope
        self.world_model_manager.set_knowledge_scope(scope)
        logger.info(f"Knowledge scope set to: {scope}")

    def process_plan(self, 
                    plan: Dict[str, Any],
                    goal_config: Optional[Dict[str, Any]],
                    task_dependencies: Optional[List],
                    area_boundaries: Optional[Dict[str, Any]],
                    category_map: Optional[Dict[str, Any]],
                    runtime_params: Optional[Dict[str, Any]]) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        Process a plan by converting dispatcher results into skill sequences.
        
        Args:
            plan: Raw plan, as dispatcher results.
            goal_config: Goal configuration.
            area_boundaries: Area boundaries.
            runtime_params: Runtime parameters.
            
        Returns:
            Skill dictionary organized by timestep.
        """
        if not plan:
            logger.error("No plan to process")
            return None
        
        try:
            dump_var("dispatcher_result", plan)  # Record raw dispatcher results.
            skills_by_timestep = self.plan_translate_coordinator.run(
                dispatcher_result=plan,
                skill_schemas=SKILL_SCHEMAS,
                dependencies = task_dependencies,
                category_map=category_map,
                goal_cfg=goal_config,
                area_boundaries=area_boundaries or {},
                runtime_params=runtime_params or {}
            )
            dump_var("skills_by_timestep", skills_by_timestep)  # Record processed skill sequences.
            return skills_by_timestep
            
        except Exception as e:
            logger.error(f"Error processing plan: {e}", exc_info=True)
            return None
    
    def process_outcomes(self, outcomes: List[Dict]):
        """Process execution results."""
        if outcomes:
            # Update goal progress.
            self.goal_progress_monitor.process_outcomes(outcomes)
    
    def is_goal_completed(self) -> bool:
        """Check whether the goal is complete."""
        goal = self.scene_graph.get_goal() if self.scene_graph else None
        if self.monitor and self.scene_graph and goal:
            goal_id = goal["id"]
            return goal_id in self.goal_progress_monitor.completed_goals
        return False
    
    def generate_terminal_feedback(self, goal_id: str, outcomes: List[Dict],
                                  plan_completed: bool, achieved: Optional[bool]) -> str:
        """Generate terminal feedback."""
        return self.goal_progress_monitor.generate_terminal_feedback(
            goal_id=goal_id,
            outcomes=outcomes,
            plan_completed=plan_completed,
            achieved=achieved
        )
    
    def get_world_model_manager(self) -> WorldModelManager:
        """Get the world model manager."""
        return self.world_model_manager
