import json
import logging
from typing import Dict, Any, List, Optional
from modules.task_solver.llm_framework.file import Logger
from modules.utils.system.root import PathManager
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.sgi_planner.prompt import (
    robot_skill_library
)
from modules.task_solver.sgi_planner.utils import transform_for_allocator
from modules.task_solver.sgi_planner.allocator import tango_allocator, convert_config_for_tango, process_allocation_to_timestep_skills
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)

class Allocation:
    def __init__(self, context: WorkflowContext, logger: Logger, path_manager: PathManager):
        self.logger = logger
        self.path_manager = path_manager
        self._context = context # Store the same shared context

    async def run(self, ready_tasks: List[Dict[str, Any]], shared_skill_groups: Optional[List] = None) -> Optional[Dict]:
        """
        Runs the task allocation phase for a given set of ready tasks.
        Returns the allocation dictionary on success, None on failure or if no tasks.
        """
        if not ready_tasks:
            dlog("No ready tasks to allocate. Skipping allocation.", logger=self.logger, level="info")
            return {} # Return empty dict for success with no action

        dlog(f"Allocating {len(ready_tasks)} ready tasks...", logger=self.logger)

        try:
            # Step 1: Translate plan for TANGO algorithm
            dlog("Alloc Step 1: Translating tasks for allocator...", logger=self.logger)
            available_robots = self._context._generated_text.get('available_robots', {})
            allocator_input = transform_for_allocator(ready_tasks, robot_skill_library, available_robots)
            self._context._generated_text["tango_translation"] = allocator_input

            # Step 2: Adjust parameters
            dlog("Alloc Step 2: Adjusting parameters for allocator...", logger=self.logger)
            real_time_pos_map = self._context._generated_text.get('real_time_pos_map', None)
            if not real_time_pos_map:
                dlog("No real-time position map found, using default.", logger=self.logger, level="error")
                return None
            
            goal_cfg = self._context._generated_text.get("goal")
            area_boundaries = self._context._generated_text.get("area_boundaries", {}) or {}
            category_map = self._context._generated_text.get("category_map", {}) or {}

            adjusted_model = convert_config_for_tango(
                allocator_input,
                real_time_pos_map,
                available_robots,
                shared_skill_groups,
                ready_tasks,
                goal_cfg=goal_cfg,
                area_boundaries=area_boundaries,
                category_map=category_map,
            )

            # Step 3: Call the TANGO Allocator
            dlog("Alloc Step 3: Running the TANGO allocator...", logger=self.logger)
            alloc_results_dir, flag_optimized, allc_data = tango_allocator(adjusted_model, self.path_manager.workspace_root)

            # Relax shared_capability_groups until allocation succeeds or the list is empty.
            while (not flag_optimized) and adjusted_model.get('shared_capability_groups'):
                dlog("Allocation failed with current constraints. Relaxing one shared capability group and retrying...", logger=self.logger, level="warning")
                removed_group = adjusted_model['shared_capability_groups'].pop()
                dlog(f"Removed constraint group: {removed_group}", logger=self.logger)
                if not adjusted_model['shared_capability_groups']:
                    dlog("All shared capability groups have been removed.", logger=self.logger)
                alloc_results_dir, flag_optimized, allc_data = tango_allocator(adjusted_model, self.path_manager.workspace_root)

            # If allocation still fails after clearing shared_capability_groups, return False.
            if not flag_optimized:
                dlog("Allocation still failed after relaxing all shared capability groups.", logger=self.logger, level="error")
                return False

            has_warning = self._post_process_and_check_warning(
                alloc_results_dir, allc_data, ready_tasks, available_robots
            )

            # If there is a warning, rerun with the same adjusted_model unchanged.
            retries_remaining = 2  
            while has_warning and retries_remaining > 0:
                dlog(f"Invalid allocation detected (warning present). Retrying with identical config... ({3 - retries_remaining}/3)", logger=self.logger, level="warning")
                alloc_results_dir, flag_optimized, allc_data = tango_allocator(adjusted_model, self.path_manager.workspace_root)
                if not flag_optimized:
                    dlog("Unexpected: allocator returned non-optimized during warning retry. Treating as failure.", logger=self.logger, level="error")
                    return False
                has_warning = self._post_process_and_check_warning(
                    alloc_results_dir, allc_data, ready_tasks, available_robots
                )
                retries_remaining -= 1

            # Treat it as success if there is no warning, or if warnings remain after 3 attempts.
            if has_warning:
                dlog("Allocation completed with warnings after attempts; treating as success per policy.", logger=self.logger, level="warning")
            else:
                dlog("Allocation successful with no warnings.", logger=self.logger, level="success")

            return True

        except Exception as e:
            dlog(f"An error occurred during task allocation: {e}", logger=self.logger, level="error")
            return False

    def _post_process_and_check_warning(
        self,
        results_dir: str,
        allocation_data: Dict[str, Any],
        ready_tasks: List[Dict[str, Any]],
        available_robots: Dict[str, Any],
    ) -> bool:
        dlog("Alloc Step 4: Post-processing results...", logger=self.logger)
        self._context._generated_text["alloc_results_dir"] = results_dir
        if allocation_data and 'result' in allocation_data:
            result_summary = allocation_data['result']
            pre_check = result_summary.get('sharedCapabilityPreCheck', 'N/A')
            energy_check = result_summary.get('energyConstraintCheck', 'N/A')
            dlog(f"  - Shared Capability Pre-check: '{pre_check}'", logger=self.logger, level="info")
            dlog(f"  - Energy Constraint Check: '{energy_check}'", logger=self.logger, level="info")

        allocation_description = process_allocation_to_timestep_skills(
            atomic_tasks=ready_tasks,
            yaml_file_path=results_dir,
            robot_inventory=available_robots,
            allocation_data=allocation_data
        )
        # Archive to context.
        self._context._generated_text["alloc_results"] = allocation_description
        dlog(f"Allocation results: {json.dumps(allocation_description['timestep_skills'], indent=2, ensure_ascii=False)}", logger=self.logger, level="info")
        return isinstance(allocation_description, dict) and ("warning" in allocation_description)
