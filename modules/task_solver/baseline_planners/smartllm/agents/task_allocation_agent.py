# -*- coding: utf-8 -*-
"""
SmartLLM Task Allocation Agent - Task allocation

Inherits ActionNode, generates a timestep-serialized execution plan
based on the decomposition and coalition results.
Output format is List[List[str]], where each sub-list is a timestep containing "robot_label:skill" pairs.
"""
import logging
from typing import Any, Dict, List, Optional

from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.baseline_planners.smartllm.prompt import (
    build_allocation_system_prompt,
    build_allocation_user_prompt,
    ALLOCATION_FEW_SHOT_EXAMPLES,
)
from modules.task_solver.baseline_planners.common.prompt_components import extract_json
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)


class TaskAllocationAgent(ActionNode):
    """Task allocation agent — outputs timestep-serialized execution plan.

    Key features:
    - Receives decomposition and coalition results, outputs timestep-organized robot:skill list
    - Supports few-shot examples
    - Parses JSON response to extract plan field

    Attributes:
        robot_labels: Available robot label list.
        use_few_shot: Whether to include few-shot examples in the prompt.
        _decomposition_result: Decomposition stage result.
        _coalition_result: Coalition stage result.
        last_allocation: Most recently parsed timestep execution plan List[List[str]].
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
            node_name="SmartLLM_Allocation",
            model_family=model_family,
            model_name_override=model_name_override,
        )
        self.robot_labels: List[str] = robot_labels or []
        self.use_few_shot: bool = use_few_shot
        self._decomposition_result: Optional[str] = None
        self._coalition_result: Optional[str] = None
        self.last_allocation: Optional[List[List[str]]] = None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self) -> None:
        """Build allocation prompt: system(role+skill library+format) + few-shot + user(decomposition+coalition)."""
        goal_type = self.context._generated_text.get("goal_type")

        system_prompt = build_allocation_system_prompt(
            robot_labels=self.robot_labels,
            goal_type=goal_type,
            use_few_shot=self.use_few_shot,
        )
        user_prompt = build_allocation_user_prompt(
            decomposition_result=self._decomposition_result or "",
            coalition_result=self._coalition_result or "",
        )

        parts = [system_prompt]

        # Few-shot examples
        if self.use_few_shot:
            for msg in ALLOCATION_FEW_SHOT_EXAMPLES:
                parts.append(f"\n{msg['content']}")

        parts.append(f"\n{user_prompt}")

        self.prompt = "\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    async def _process_response(self, content: str) -> str:
        """Parse JSON response, extract timestep-serialized execution plan.

        plan is a list where each element is a timestep (sub-list),
        and each sub-list element is a "robot_label:skill" string.

        Args:
            content: Raw LLM response text.

        Returns:
            Raw response text (parsed result stored in self.last_allocation).

        Raises:
            ValueError: On JSON parse failure or missing required fields.
        """
        parsed = extract_json(content)

        plan = parsed.get("plan")
        if plan is None:
            raise ValueError(
                "Allocation response missing 'plan' field. "
                f"Parsed keys: {list(parsed.keys())}"
            )
        if not isinstance(plan, list):
            raise ValueError(
                f"'plan' must be a list of timesteps, got {type(plan).__name__}"
            )

        valid_labels = set(self.robot_labels) if self.robot_labels else None
        normalized: List[List[str]] = []

        for ts_idx, timestep in enumerate(plan):
            if not isinstance(timestep, list):
                raise ValueError(
                    f"Timestep {ts_idx} must be a list, got {type(timestep).__name__}"
                )
            ts_pairs: List[str] = []
            for entry in timestep:
                entry_str = str(entry).strip()
                # Validate "robot_label:skill" format
                if ":" not in entry_str:
                    logger.warning(
                        f"Ignoring malformed entry '{entry_str}' in timestep {ts_idx} "
                        f"(expected 'robot_label:skill' format)"
                    )
                    continue
                robot_label, skill = entry_str.split(":", 1)
                robot_label = robot_label.strip()
                skill = skill.strip()
                if valid_labels and robot_label not in valid_labels:
                    logger.warning(
                        f"Ignoring unknown robot label '{robot_label}' in timestep {ts_idx}"
                    )
                    continue
                ts_pairs.append(f"{robot_label}:{skill}")
            if ts_pairs:
                normalized.append(ts_pairs)

        self.last_allocation = normalized
        return content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_inputs(
        self,
        decomposition_result: str,
        coalition_result: str,
    ) -> None:
        """Set results from the two previous stages.

        Args:
            decomposition_result: TaskDecompositionAgent output (JSON string or text).
            coalition_result: CoalitionFormationAgent output (text).
        """
        self._decomposition_result = decomposition_result
        self._coalition_result = coalition_result

    def reset(self) -> None:
        """Reset agent state."""
        self._decomposition_result = None
        self._coalition_result = None
        self.last_allocation = None
