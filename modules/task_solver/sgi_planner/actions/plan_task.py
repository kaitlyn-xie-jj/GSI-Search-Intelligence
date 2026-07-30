import os
import json
from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.llm_framework.core.parser import *
from modules.utils.system.root import PathManager
from datetime import datetime
from tqdm.asyncio import tqdm
from modules.task_solver.sgi_planner.prompt import (
    robot_skill_library
)
from modules.task_solver.sgi_planner.utils import analyze_task_dependencies, validate_complete, to_concise_robot_info
from modules.task_solver.sgi_planner.prompt.runtime_builders import (
    compose_master_context, select_prompt_and_feedback
)
from modules.utils.system.logging_utils import dlog
from modules.utils.system.var_dump import dump_var
from modules.utils.replan_recorder import ReplanDatasetRecorder
from modules.config.system_config import config as system_config


class TaskPlan(ActionNode):
    def __init__(self, logger: Logger, context: WorkflowContext, 
                 next_text="", node_name="", 
                 model_family: str = None, model_name_override: str = None,
                 planner_mode: str = "phase", use_environment_model: bool = False,
                 validate_plan: bool = True):
        super().__init__(logger, context, next_text, node_name, 
                        model_family=model_family, model_name_override=model_name_override)
        self.planner_mode = planner_mode
        self.use_environment_model = use_environment_model
        self.validate_plan = validate_plan

    @staticmethod
    def _format_feedback_compact(feedback_list):
        """Format a feedback data list as compact multiline text."""
        blocks = []
        for fb in feedback_list:
            lines = []
            if fb.get("type"):
                lines.append(f"type: {fb['type']}")
            if fb.get("reason"):
                lines.append(f"reason: {fb['reason']}")
            if fb.get("failed_skills"):
                lines.append("failed_skills:")
                for s in fb["failed_skills"]:
                    lines.append(f"  {s}")
            if fb.get("completed_skills"):
                lines.append("completed_skills:")
                for s in fb["completed_skills"]:
                    lines.append(f"  {s}")
            if fb.get("user_feedback"):
                lines.append(f"user_feedback: {fb['user_feedback']}")
            blocks.append("\n".join(lines))
        return "\n---\n".join(blocks)

    def _build_prompt(self):
        instruction = self.context._generated_text.get('instruction', '')
        scene_desc = self.context._generated_text.get("scene_desc")
        goal_type = self.context._generated_text.get("goal_type")
        available_robots = self.context._generated_text.get('available_robots', {})
        feedback = self.context._generated_text.get('feedback_data', None)
        previous_plan =self.context._generated_text.get('task_plan_brief', None) or self.context._generated_text.get('task_plan', None)
        is_replanning = self.context._generated_text.get('is_replanning', False)

        available_robots = to_concise_robot_info(available_robots)
        template, feedback_context = select_prompt_and_feedback(
            planner_mode=self.planner_mode,
            use_separate_prompts=True,
            is_replanning=is_replanning,
        )

        master_context = compose_master_context(
            planner_mode=self.planner_mode,
            use_environment_model=self.use_environment_model,
            scene_desc=scene_desc,
            goal_type=goal_type,
            is_replanning=is_replanning,
        )

        feedback_context_section = ""
        if feedback:
            feedback_str = self._format_feedback_compact(feedback)
            feedback_context_section = feedback_context.format(
                feedback_str=feedback_str
            )

        self.prompt = None
        self.prompt = template.format(
            master_context=master_context,
            available_robots=available_robots,
            feedback_context_section=feedback_context_section,
            instruction=instruction,
            previous_plan=previous_plan,
        )

    async def run(self, auto_next: bool = True):
        """
        - Initial planning: use the parent logic directly to build the prompt, call the LLM, and call _process_response.
        - save_llm_io=True: write llm_trace for every TaskPlan call; when replan collection is hit, write replan_records after the LLM returns.
        - save_llm_io=False: keep the old prompt-only collection and stop_after_record behavior.
        """
        if not ReplanDatasetRecorder.is_enabled() or self.context is None:
            return await super().run(auto_next=auto_next)

        gt = getattr(self.context, "_generated_text", {})
        is_replanning = bool(gt.get("is_replanning", False))

        if not ReplanDatasetRecorder.save_llm_io_enabled():
            if not is_replanning:
                return await super().run(auto_next=auto_next)
            self._build_prompt()
            prompt = getattr(self, "prompt", "")
            ReplanDatasetRecorder.handle_replan_prompt(self.context, prompt)
            return await super().run(auto_next=auto_next)

        self._build_prompt()
        prompt = getattr(self, "prompt", "")
        if is_replanning:
            ReplanDatasetRecorder.prepare_replan_prompt(self.context, prompt)

        self.logger.log(f"Action: {str(self)}", level="action")
        try:
            result = await self._run()
        except Exception as exc:
            gt = getattr(self.context, "_generated_text", {})
            response = gt.get("planner_response")
            error = repr(exc)
            ReplanDatasetRecorder.record_llm_call(
                self.context,
                node_name=str(self),
                prompt=prompt,
                response=response,
                is_replanning=is_replanning,
                error=error,
            )
            if is_replanning:
                ReplanDatasetRecorder.finalize_replan_prompt(
                    self.context,
                    response=response,
                    error=error,
                )
            raise

        gt = getattr(self.context, "_generated_text", {})
        response = gt.get("planner_response", "")
        ReplanDatasetRecorder.record_llm_call(
            self.context,
            node_name=str(self),
            prompt=prompt,
            response=response,
            is_replanning=is_replanning,
        )
        if is_replanning:
            ReplanDatasetRecorder.finalize_replan_prompt(self.context, response=response)
        if auto_next and self._next is not None:
            return await self._next.run()
        return result

    async def _process_response(self, response: str) -> str:
        # Record the raw response and prompt for external persistence.
        self.context._generated_text["planner_prompt"] = getattr(self, "prompt", "")
        self.context._generated_text["planner_response"] = response
        dump_var("prompt", getattr(self, "prompt", ""))
        dump_var("response", response)
        
        # Parse the response.
        results_content = parse_text(response, "json", True)

        if len(results_content) > 0:
            plan_str = results_content[0]

            if self.validate_plan:
                # Enable validation, including automatic fixes.
                results = validate_complete(plan_str, robot_skill_library)
            
                if not results['overall_valid']:
                    error_msgs = []
                    for level, result in results.items():
                        if isinstance(result, dict) and not result.get('valid', True):
                            error_msgs.extend(result.get('errors', []))
                    unique_errors = list(set(str(e) for e in error_msgs))
                    raise ValueError(f"Task validation failed: {'; '.join(unique_errors)}")
            
                # Validation succeeded; update the context with fixed data.
                if results['fixed_data']:
                    self.context._generated_text["task_plan"] = results['fixed_data']
                else:
                    self.context._generated_text["task_plan"] = plan_str
            else:
                # Validation is disabled; use the plan directly.
                self.context._generated_text["task_plan"] = plan_str
        
        dlog(f"Output Task Plan Success", logger=self.logger, level="success")

async def process_single_query(i, semaphore):
    """
    Process a single query with concurrency control
    """
    async with semaphore:  # limit concurrent access
        curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path_manager = PathManager(
            base_results_dir=os.path.dirname(os.path.realpath(__file__)) + f"/../results/{curr_run}",
        )

        log_file_path = path_manager.workspace_root / "log.md"
        logger = Logger(log_file_path=str(log_file_path))

        critic = TaskPlan(logger=logger)
        await critic.run()

async def ask_llm_concurrent(args):
    """
    Main function to run the LLM queries concurrently
    """
    # Create a semaphore to limit concurrent access
    semaphore = asyncio.Semaphore(30)

    # Create a progress bar
    task_num = 1
    progress_bar = tqdm(total=task_num, desc="Processing batches", unit="batch")

    # Create all tasks
    tasks = [
        process_single_query(i, semaphore)
        for i in range(task_num)
    ]

    # Process tasks concurrently
    for idx, task_future in enumerate(asyncio.as_completed(tasks)):
        result = await task_future
        progress_bar.update(1)

    # Complete the progress bar display
    progress_bar.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(ask_llm_concurrent(None))
