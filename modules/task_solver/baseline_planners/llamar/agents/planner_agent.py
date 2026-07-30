# -*- coding: utf-8 -*-
"""
LLaMAR Planner Agent - Generate and update subtask lists

Inherits ActionNode, uses Planner prompt template to build prompts,
parses LLM response into plan list and reason string.
"""
from typing import Any, Dict, List, Optional

from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.baseline_planners.llamar.prompt.planner_prompt import (
    build_planner_system_prompt,
    build_planner_user_prompt,
    PLANNER_FEW_SHOT_EXAMPLES,
)
from modules.task_solver.baseline_planners.common.prompt_components import extract_json
from modules.task_solver.baseline_planners.llamar.prompt.prompt_utils import (
    format_subtasks,
)
from modules.task_solver.sgi_planner.prompt.observation import format_observation


class PlannerAgent(ActionNode):
    """LLaMAR Planner Agent — generates/updates subtask lists.

    Uses WorldModelManager's known_nodes/known_edges as observation input,
    combined with current open/closed subtasks, calls LLM to generate a new plan.

    Attributes:
        robot_labels: Available robot labels in the current scene.
        known_nodes: WorldModelManager node list.
        known_edges: WorldModelManager edge list.
        open_subtasks: Current pending subtask list.
        closed_subtasks: Current completed subtask list.
        memory: Accumulated scene memory string.
        last_plan: Most recently parsed plan list.
        last_reason: Most recently parsed reason string.
    """

    def __init__(
        self,
        logger: Logger,
        context: WorkflowContext,
        robot_labels: Optional[List[str]] = None,
        node_name: str = "LLaMAR_Planner",
        model_family: str = None,
        model_name_override: str = None,
        use_few_shot: bool = False,
    ):
        super().__init__(
            logger=logger,
            context=context,
            next_text="",
            node_name=node_name,
            model_family=model_family,
            model_name_override=model_name_override,
        )
        self.robot_labels: List[str] = robot_labels or []
        self.known_nodes: List[Dict] = []
        self.known_edges: List[Dict] = []
        self.open_subtasks: Optional[List[str]] = None
        self.closed_subtasks: Optional[List[str]] = None
        self.memory: str = ""
        
        # Few-shot config
        self.use_few_shot = use_few_shot

        # Parsed result cache
        self.last_plan: Optional[List[str]] = None
        self.last_reason: str = ""

    # ------------------------------------------------------------------
    # ActionNode interface
    # ------------------------------------------------------------------

    def _build_prompt(self) -> None:
        """Build the full prompt for Planner Agent.

        Gets instruction and goal_type from context._generated_text,
        combined with current observation and subtask state.
        Uses unified prompt structure: system + few-shot + user.
        """
        instruction = self.context._generated_text.get("instruction", "")
        goal_type = self.context._generated_text.get("goal_type")

        observation = format_observation(
            self.known_nodes, self.known_edges, self.robot_labels
        )
        subtasks_text = format_subtasks(self.open_subtasks, self.closed_subtasks)

        system_prompt = build_planner_system_prompt(
            goal_type=goal_type,
        )
        user_prompt = build_planner_user_prompt(
            instruction=instruction,
            observation=observation,
            subtasks_text=subtasks_text,
            memory=self.memory,
        )

        parts = [system_prompt]

        # Few-shot examples
        if self.use_few_shot:
            for msg in PLANNER_FEW_SHOT_EXAMPLES:
                parts.append(f"\n{msg['content']}")

        parts.append(f"\n{user_prompt}")

        self.prompt = "\n".join(parts)

    async def _process_response(self, content: str) -> str:
        """Parse LLM response into plan list and reason string.

        Attempts to extract JSON from response, supports two formats:
        1. Wrapped in ```json ... ``` code blocks
        2. Direct JSON text

        Args:
            content: Raw LLM response text.

        Returns:
            Raw response text (parsed results stored in self.last_plan / self.last_reason).

        Raises:
            ValueError: On JSON parse failure or missing required fields,
                        triggers ActionNode retry.
        """
        parsed = extract_json(content)

        plan = parsed.get("plan")
        reason = parsed.get("reason", "")

        if plan is None:
            raise ValueError(
                "Planner response missing 'plan' field. "
                f"Parsed keys: {list(parsed.keys())}"
            )
        if not isinstance(plan, list):
            raise ValueError(
                f"Planner 'plan' field must be a list, got {type(plan).__name__}"
            )

        self.last_plan = [str(s) for s in plan]
        self.last_reason = str(reason)

        # Also write to context for other modules to read
        self.context._generated_text["llamar_plan"] = self.last_plan
        self.context._generated_text["llamar_plan_reason"] = self.last_reason

        return content

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def set_state(
        self,
        known_nodes: List[Dict],
        known_edges: List[Dict],
        open_subtasks: Optional[List[str]],
        closed_subtasks: Optional[List[str]],
        memory: str = "",
    ) -> None:
        """Set all input state needed for this round at once."""
        self.known_nodes = known_nodes
        self.known_edges = known_edges
        self.open_subtasks = open_subtasks
        self.closed_subtasks = closed_subtasks
        self.memory = memory
