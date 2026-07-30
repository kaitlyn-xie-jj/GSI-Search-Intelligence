import json
from modules.task_solver.llm_framework.file import File, Logger
from modules.utils.system.root import PathManager
from modules.task_solver.llm_framework.core.action import *
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.sgi_planner.actions import *
from modules.task_solver.sgi_planner.prompt import *
from modules.utils.system.logging_utils import dlog

class PlanGeneration:
    def __init__(self, context: WorkflowContext, logger: Logger, 
                 path_manager: PathManager, planner_mode: str = "phase", 
                 use_environment_model: bool = False, validate_plan: bool = True):
        self.logger = logger
        self.path_manager = path_manager
        self._context = context  # Store the shared context
        
        # Initialize agents with model type
        self.planner = TaskPlan(logger=self.logger, context=self._context, 
                               planner_mode=planner_mode, use_environment_model=use_environment_model,
                               validate_plan=validate_plan)

    async def run(self):
        # Decomp Step 1: Initial planning
        dlog(f"Decomp Step 1: Initial planning", logger=self.logger)
        await self.planner.run()

        current_plan = self._context._generated_text.get("task_plan", None)
        if current_plan is None:
            dlog("Initial plan generation failed.", logger=self.logger, level="error")
            return False

        # Decomp Step 2: Output
        # dlog("Decomp Step 2: Generating final output", logger=self.logger)
        # await self._generate_final_output(current_plan, 1)

        return True

    async def _generate_final_output(self, final_plan, iterations_used):
        """Generate final workflow output and documentation"""
        
        dlog(f"Workflow completed after {iterations_used} iteration(s)", logger=self.logger)
        
        # Create the documentation content
        flow_text = f"```json\n{final_plan}\n```"
        
        try:
            # 1. Provide the root path from the instance's path_manager
            flow_file = File(name="flow.md", root=self.path_manager.workspace_root)
            
            # 2. Call the write method
            flow_file.write(flow_text)
            
            # 3. The workflow itself logs the success
            dlog(f"Final flow documentation written to: {flow_file.file_path}", logger=self.logger, level="info")

        except Exception as e:
            # 4. The workflow catches exceptions and logs the error
            dlog(f"Failed to write flow documentation: {e}", logger=self.logger, level="error")