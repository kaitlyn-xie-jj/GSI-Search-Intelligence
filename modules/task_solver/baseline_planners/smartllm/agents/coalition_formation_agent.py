# -*- coding: utf-8 -*-
"""
SmartLLM Coalition Formation Agent - Coalition formation

Inherits ActionNode, determines robot coalitions for each sub-task
based on the decomposition and robot capabilities.
Outputs coalition text, passed directly to TaskAllocationAgent.
"""
import logging
from typing import Dict, List, Optional

from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.baseline_planners.smartllm.prompt import (
    build_coalition_system_prompt,
    build_coalition_user_prompt,
    COALITION_FEW_SHOT_EXAMPLES,
)
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)


class CoalitionFormationAgent(ActionNode):
    """Coalition formation agent — determines robot coalitions for each sub-task.

    Key features:
    - Receives decomposition, outputs coalition assignment text
    - Supports few-shot examples
    - Output is text format, passed directly to downstream TaskAllocationAgent

    Attributes:
        robot_labels: Available robot label list.
        use_few_shot: Whether to include few-shot examples in the prompt.
        _decomposition_result: Decomposition result from the previous stage.
        last_coalition: Most recent coalition text.
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
            node_name="SmartLLM_Coalition",
            model_family=model_family,
            model_name_override=model_name_override,
        )
        self.robot_labels: List[str] = robot_labels or []
        self.use_few_shot: bool = use_few_shot
        self.known_nodes: List[Dict] = []
        self.known_edges: List[Dict] = []
        self._decomposition_result: Optional[str] = None
        self.last_coalition: Optional[str] = None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self) -> None:
        """Build coalition prompt: system(role+skill library+rules) + few-shot + user(env+decomposition)."""
        system_prompt = build_coalition_system_prompt(
            robot_labels=self.robot_labels,
            use_few_shot=self.use_few_shot,
        )
        user_prompt = build_coalition_user_prompt(
            known_nodes=self.known_nodes,
            known_edges=self.known_edges,
            robot_labels=self.robot_labels,
            decomposition_result=self._decomposition_result or "",
        )

        parts = [system_prompt]

        # Few-shot examples
        if self.use_few_shot:
            for msg in COALITION_FEW_SHOT_EXAMPLES:
                parts.append(f"\n{msg['content']}")

        parts.append(f"\n{user_prompt}")

        self.prompt = "\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    async def _process_response(self, content: str) -> str:
        """Parse text response, extract coalition assignment.

        Coalition is in text format, stored directly.

        Args:
            content: Raw LLM response text.

        Returns:
            Raw response text.
        """
        self.last_coalition = content.strip()
        return content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_decomposition(self, decomposition_result: str) -> None:
        """Set decomposition result from the previous stage.

        Args:
            decomposition_result: TaskDecompositionAgent output (JSON string or text).
        """
        self._decomposition_result = decomposition_result

    def reset(self) -> None:
        """Reset agent state."""
        self._decomposition_result = None
        self.last_coalition = None
