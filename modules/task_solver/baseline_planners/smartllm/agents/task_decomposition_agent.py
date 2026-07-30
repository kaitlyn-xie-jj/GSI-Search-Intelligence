# -*- coding: utf-8 -*-
"""
SmartLLM Task Decomposition Agent - Task decomposition

Inherits ActionNode, decomposes user instructions into a sub-task
list with parameterized skills.
Outputs general decomposition (sub_tasks + execution_order), no code.
"""
import logging
from typing import Any, Dict, List, Optional

from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.baseline_planners.smartllm.prompt import (
    build_decomposition_system_prompt,
    build_decomposition_user_prompt,
    DECOMPOSITION_FEW_SHOT_EXAMPLES,
)
from modules.task_solver.baseline_planners.common.prompt_components import (
    extract_json,
)
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)



class TaskDecompositionAgent(ActionNode):
    """Task decomposition agent -- outputs general decomposition (no code).

    Key features:
    - Decomposes user instructions into sub-task list with parameterized skills
    - Supports few-shot examples and replanning feedback
    - Parses JSON response to extract sub_tasks and execution_order

    Attributes:
        robot_labels: Available robot label list.
        use_few_shot: Whether to include few-shot examples in the prompt.
        _feedback_text: Feedback text for replanning.
        last_decomposition: Most recently parsed decomposition result.
    """

    def __init__(
        self,
        logger_: Logger,
        context: WorkflowContext,
        robot_labels: Optional[List[str]] = None,
        model_family: str = None,
        model_name_override: str = None,
        use_few_shot: bool = False,
    ):
        super().__init__(
            logger=logger_,
            context=context,
            next_text="",
            node_name="SmartLLM_Decomposition",
            model_family=model_family,
            model_name_override=model_name_override,
        )
        self.robot_labels: List[str] = robot_labels or []
        self.use_few_shot: bool = use_few_shot
        self.known_nodes: List[Dict] = []
        self.known_edges: List[Dict] = []
        self._feedback_text: Optional[str] = None
        self.last_decomposition: Optional[Dict[str, Any]] = None

    def _build_prompt(self) -> None:
        """Build decomposition prompt: system + few-shot + user."""
        instruction = self.context._generated_text.get("instruction", "")
        goal_type = self.context._generated_text.get("goal_type")

        system_prompt = build_decomposition_system_prompt(
            robot_labels=self.robot_labels,
            goal_type=goal_type,
            use_few_shot=self.use_few_shot,
        )
        user_prompt = build_decomposition_user_prompt(
            instruction=instruction,
            known_nodes=self.known_nodes,
            known_edges=self.known_edges,
            robot_labels=self.robot_labels,
            feedback_text=self._feedback_text,
        )

        parts = [system_prompt]

        if self.use_few_shot:
            for msg in DECOMPOSITION_FEW_SHOT_EXAMPLES:
                parts.append(f"\n{msg['content']}")

        parts.append(f"\n{user_prompt}")
        self.prompt = "\n".join(parts)

    async def _process_response(self, content: str) -> str:
        """Parse JSON response, extract general decomposition.

        Args:
            content: Raw LLM response text.

        Returns:
            Raw response text (result stored in self.last_decomposition).

        Raises:
            ValueError: On JSON parse failure or missing required fields.
        """
        parsed = extract_json(content)

        sub_tasks = parsed.get("sub_tasks")
        if sub_tasks is None:
            raise ValueError(
                "Decomposition response missing 'sub_tasks' field. "
                f"Parsed keys: {list(parsed.keys())}"
            )
        if not isinstance(sub_tasks, list):
            raise ValueError(
                f"'sub_tasks' must be a list, got {type(sub_tasks).__name__}"
            )

        for i, st in enumerate(sub_tasks):
            if not isinstance(st, dict):
                raise ValueError(
                    f"sub_tasks[{i}] must be a dict, got {type(st).__name__}"
                )
            if "name" not in st:
                raise ValueError(f"sub_tasks[{i}] missing 'name' field")
            if "skills_required" not in st:
                raise ValueError(f"sub_tasks[{i}] missing 'skills_required' field")
            if not isinstance(st["skills_required"], list):
                raise ValueError(
                    f"sub_tasks[{i}]['skills_required'] must be a list, "
                    f"got {type(st['skills_required']).__name__}"
                )

        execution_order = parsed.get("execution_order", "sequential")

        self.last_decomposition = {
            "sub_tasks": sub_tasks,
            "execution_order": execution_order,
        }
        return content

    def set_feedback(self, feedback_text: str) -> None:
        """Set replanning feedback text.

        Args:
            feedback_text: Feedback text formatted by FeedbackProcessor.
        """
        self._feedback_text = feedback_text

    def reset(self) -> None:
        """Reset agent state."""
        self._feedback_text = None
        self.last_decomposition = None
