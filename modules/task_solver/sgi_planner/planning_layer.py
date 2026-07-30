import time
import json
from typing import Dict, Optional, Any, List, Tuple

from .plan_module import PlanGeneration
from .alloc_module import Allocation
from .task_graph_manager import TaskGraphManager
from ..llm_framework.core.context import WorkflowContext
from .feedback_processor import FeedbackProcessor, ReplanningStrategy
from ...utils.system.logging_utils import dlog

class PlanningLayer:
    """
    Planning Layer.
    - Manages the entire planning process internally.
    - Uses a TaskGraphManager to handle complex, multi-stage plans.
    - Calls allocation as a service for ready tasks.
    """
    def __init__(self, logger, path_manager, planner_mode: str,
                 use_environment_model: bool, robot_type_list: list,
                 world_model, context: WorkflowContext,
                 max_steps: int = 15, validate_plan: bool = True):
        self.logger = logger
        self.path_manager = path_manager
        self.context = context
        self.world_model = world_model
        self.planner_mode = planner_mode
        self.max_steps = max_steps
        
        # Internal modules
        self.plan_generator = PlanGeneration(
            context=self.context, logger=self.logger, path_manager=self.path_manager,
            planner_mode=self.planner_mode, use_environment_model=use_environment_model,
            validate_plan=validate_plan
        )
        self.allocation_module = Allocation(
            context=self.context, logger=self.logger, path_manager=self.path_manager
        )

        # State management
        self.task_graph_manager: Optional[TaskGraphManager] = None

        self.feedback_processor = FeedbackProcessor(
            logger=self.logger,
            context=self.context
        )

        # Review callbacks injected by the solver; default behavior passes through.
        self.review_task_graph = None
        self.review_skill_list = None

        # Pre-replanning callback injected by the solver to refresh feedback_data and context.
        self.prepare_full_replan = None

        self._no_ready_tasks_streak: int = 0  # Consecutive empty ready_tasks replan count.

        self._metrics = {
            "llm_calls": 0, "llm_durations": [], "allocation_durations": [], "batch_count": 0,
            "replans_full": 0, "replans_partial": 0, "planning_durations": [],
        }

    async def generate_plan(self) -> Optional[Dict]:
        """
        Main entry point for the planning layer (with timing).
        - On FULL replan, generates a new graph.
        - On PARTIAL or NONE, processes outcomes and finds the next executable batch.
        """
        _t0 = time.time()
        result = await self._generate_plan_inner()
        self._metrics["planning_durations"].append(round(time.time() - _t0, 6))
        return result

    async def _generate_plan_inner(self) -> Optional[Dict]:
        """Actual generate_plan logic."""
        replanning_strategy = self.feedback_processor.replanning_strategy
        
        # Record replanning metrics.
        if replanning_strategy == ReplanningStrategy.FULL:
            if self._metrics["llm_calls"] > 0:
                # Non-initial planning counts as one full replan.
                self._metrics["replans_full"] += 1
        elif replanning_strategy == ReplanningStrategy.PARTIAL:
            self._metrics["replans_partial"] += 1
        
        if replanning_strategy == ReplanningStrategy.FULL:
            if self.context and hasattr(self.context, '_generated_text'):
                runtime_params = self.context._generated_text.get("runtime_params")
                if runtime_params:
                    self.context._generated_text["runtime_params"] = {}
            return await self._generate_new_task_graph()
        else:
            if not self.task_graph_manager:
                dlog("Cannot do partial plan, no active task graph. Forcing FULL replan.", logger=self.logger, level="warning")
                return await self._generate_new_task_graph()
            
            return await self._get_next_executable_batch()

    async def _generate_new_task_graph(self) -> Optional[Dict]:
        """Generates a new plan from the LLM and gets the first batch of tasks."""
        dlog("Generating new task graph via LLM...", logger=self.logger, level="stage")
        self.task_graph_manager = None # Clear previous graph

        # Run LLM to get the plan string
        t0 = time.time()
        await self.plan_generator.run() 
        validated_plan = self.context._generated_text.get("task_plan", None)
        if not validated_plan:
            dlog("LLM agent (TaskPlan) failed to generate a valid plan string.", logger=self.logger, level="error")
            return None
        self._metrics["llm_calls"] += 1
        self._metrics["llm_durations"].append(round(time.time() - t0, 6))

        # HITL review: task graph review.
        if self.review_task_graph:
            validated_plan = await self.review_task_graph(validated_plan)
        if not validated_plan:
            dlog("Task graph review failed or returned empty.", logger=self.logger, level="error")
            return None

        try:
            # Initialize the graph manager with the new plan
            plan_data = json.loads(validated_plan) if isinstance(validated_plan, str) else validated_plan
            self.task_graph_manager = TaskGraphManager(plan_data, self.logger, self.context, self.world_model, self.planner_mode)
            dlog("Task graph initialized successfully.", logger=self.logger)
        except ValueError as e:
            dlog(f"Failed to initialize task graph: {e}", logger=self.logger, level="error")
            return None
        
        # Get and allocate the first batch of ready tasks
        return await self._get_next_executable_batch()

    async def _get_next_executable_batch(self) -> Optional[Dict]:
        """
        Finds the next ready tasks, and allocates them.
        """
        self._metrics["batch_count"] += 1

        if not self.task_graph_manager:
            return None

        # Check if the entire mission is complete
        if self.task_graph_manager.is_complete():
            dlog("Task graph is complete. No more tasks to allocate.", logger=self.logger, level="success")
            return {}

        # Robot fault recovery path: restore from snapshot and trim.
        if self.task_graph_manager._robot_fault_recovery:
            ready_tasks, dispatchable_groups = self.task_graph_manager.get_ready_tasks_for_recovery()
        else:
            # Get the next set of ready tasks
            ready_tasks = self.task_graph_manager.get_ready_tasks()
            dispatchable_groups = (
                self.task_graph_manager.get_dispatchable_skill_groups(ready_tasks)
                if ready_tasks else []
            )

        if not ready_tasks:
            return {}

        # Valid tasks exist, so reset the consecutive empty count.
        self._no_ready_tasks_streak = 0

        # Allocate the ready tasks
        t1 = time.time()
        await self.allocation_module.run(ready_tasks=ready_tasks, shared_skill_groups=dispatchable_groups)
        allocation_plan = self.context._generated_text.get("alloc_results", None)
        self._metrics["allocation_durations"].append(round(time.time() - t1, 6))

        if allocation_plan is None:
            dlog("Allocation module failed for ready tasks.", logger=self.logger, level="error")
            return None # Propagate failure

        # HITL review: skill allocation review.
        if self.review_skill_list:
            allocation_plan = await self.review_skill_list(allocation_plan)
        if allocation_plan is None:
            dlog("Skill list review failed or returned empty.", logger=self.logger, level="error")
            return None

        # 5. Mark allocated tasks as dispatched
        dispatched_ids = [task['task_id'] for task in ready_tasks]
        self.task_graph_manager.mark_as_dispatched(dispatched_ids)

        # 6. Save allocation snapshot for robot fault recovery.
        self.task_graph_manager.save_allocation_snapshot(ready_tasks, dispatchable_groups)
        
        return allocation_plan
    
    def update_graph_from_feedback(self):
        """
        Public method to allow external callers to update the graph state based on the latest feedback.
        """
        if self.task_graph_manager:
            self.task_graph_manager.update_from_feedback(self.feedback_processor)

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics, including task graph completion statistics."""
        metrics = {
            "llm_calls": self._metrics["llm_calls"],
            "llm_durations": list(self._metrics["llm_durations"]),
            "allocation_durations": list(self._metrics["allocation_durations"]),
            "batch_count": self._metrics["batch_count"],
            "replans_full": self._metrics["replans_full"],
            "replans_partial": self._metrics["replans_partial"],
            "planning_durations": list(self._metrics["planning_durations"]),
        }
        
        # Task graph completion statistics.
        if self.task_graph_manager:
            stats = self.task_graph_manager.get_completion_stats()
            completed = stats.get("completed", 0)
            total = stats.get("total", 0)
            metrics["completed_tasks"] = completed
            metrics["total_tasks"] = total
            metrics["atomic_task_completion_rate"] = round(completed / total, 4) if total else 0.0
        
        return metrics
    
    def get_goal_config(self) -> Optional[Dict]:
        """Get goal configuration."""
        goal = self.context._generated_text.get('goal')
        if goal:
            return getattr(goal, 'config', None) or goal
        return None
    
    def get_area_boundaries(self) -> Dict:
        """Get area boundaries."""
        area_boundaries = self.context._generated_text.get('area_boundaries', {})
        return area_boundaries or {}
    
    def get_category_map(self) -> Dict:
        """Get category mapping."""
        category_map = self.context._generated_text.get('category_map', {})
        return category_map or {}
    
    def get_runtime_params(self) -> Dict:
        """Get runtime parameters."""
        runtime_params = self.context._generated_text.get('runtime_params', {})
        return runtime_params or {}
    
    def get_dependencies(self) -> Dict:
        if self.task_graph_manager:
            return self.task_graph_manager.get_ready_tasks_dependencies()
        return {}
    
    def cleanup(self):
        """Clean up resources."""
        self.task_graph_manager = None
        dlog("PlanningLayer cleaned up.", logger=self.logger)

    def reset(self):
        """Reset planning layer state before a new task starts."""
        self.task_graph_manager = None
        self.feedback_processor.reset()
        self._no_ready_tasks_streak = 0
        self._metrics = {
            "llm_calls": 0, "llm_durations": [], "allocation_durations": [], "batch_count": 0,
            "replans_full": 0, "replans_partial": 0, "planning_durations": [],
        }
