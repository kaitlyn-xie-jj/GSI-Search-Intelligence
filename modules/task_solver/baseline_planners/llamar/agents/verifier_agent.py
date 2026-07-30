# -*- coding: utf-8 -*-
"""
LLaMAR Verifier Agent - Determine subtask completion status

Inherits ActionNode, uses Verifier prompt template to build prompts,
parses LLM response into completed subtask list and reason string.
"""
from typing import Any, Dict, List, Optional

from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.baseline_planners.llamar.prompt.verifier_prompt import (
    build_verifier_system_prompt,
    build_verifier_user_prompt,
)
from modules.task_solver.baseline_planners.common.prompt_components import extract_json
from modules.task_solver.baseline_planners.llamar.prompt.prompt_utils import (
    format_subtasks,
    format_feedback,
    format_previous_actions,
)
from modules.task_solver.sgi_planner.prompt.observation import format_observation


class VerifierAgent(ActionNode):
    """LLaMAR Verifier Agent — determines which subtasks are completed.

    Based on observation, subtask list, memory, and current step execution feedback,
    calls LLM to determine which subtasks can be moved from open to closed.

    Attributes:
        robot_labels: Available robot labels in the current scene.
        known_nodes: WorldModelManager node list.
        known_edges: WorldModelManager edge list.
        open_subtasks: Current pending subtask list.
        closed_subtasks: Current completed subtask list.
        memory: Accumulated scene memory string.
        per_robot_feedback: Per-robot feedback dictionary for the current step.
        last_completed: Most recently parsed completed subtask list.
        last_reason: Most recently parsed reason string.
    """

    def __init__(
        self,
        logger: Logger,
        context: WorkflowContext,
        robot_labels: Optional[List[str]] = None,
        node_name: str = "LLaMAR_Verifier",
        model_family: str = None,
        model_name_override: str = None,
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
        self.active_robot_labels: List[str] = []
        self.known_nodes: List[Dict] = []
        self.known_edges: List[Dict] = []
        self.open_subtasks: Optional[List[str]] = None
        self.closed_subtasks: Optional[List[str]] = None
        self.memory: str = ""
        self.per_robot_feedback: Dict[str, str] = {}
        self.previous_actions: Dict[str, str] = {}
        self.previous_successes: Dict[str, bool] = {}

        # Parsed result cache
        self.last_completed: List[str] = []
        self.last_reason: str = ""

    # ------------------------------------------------------------------
    # ActionNode interface
    # ------------------------------------------------------------------

    def _build_prompt(self) -> None:
        """Build the full prompt for Verifier Agent.

        Includes seven input components: instruction, observation, previous actions
        (active robots only), subtasks, memory, feedback.
        """
        instruction = self.context._generated_text.get("instruction", "")

        observation = format_observation(
            self.known_nodes, self.known_edges, self.robot_labels
        )
        subtasks_text = format_subtasks(self.open_subtasks, self.closed_subtasks)
        feedback_text = format_feedback(self.per_robot_feedback)
        prev_actions_text = format_previous_actions(
            self.previous_actions,
            self.previous_successes,
            self.active_robot_labels,
        )

        system_prompt = build_verifier_system_prompt(self.active_robot_labels)
        user_prompt = build_verifier_user_prompt(
            instruction=instruction,
            observation=observation,
            subtasks_text=subtasks_text,
            memory=self.memory,
            feedback=feedback_text,
            previous_actions=prev_actions_text,
        )

        self.prompt = f"{system_prompt}\n\n{user_prompt}"

    async def _process_response(self, content: str) -> str:
        """Parse LLM response into completed subtask list and reason.

        Expected JSON fields:
        - "completed subtasks": List[str]
        - "reason": str

        Args:
            content: Raw LLM response text.

        Returns:
            Raw response text.

        Raises:
            ValueError: On JSON parse failure or field type error,
                        triggers ActionNode retry.
        """
        parsed = extract_json(content)

        completed = parsed.get("completed subtasks")
        reason = parsed.get("reason", "")

        if completed is None:
            raise ValueError(
                "Verifier response missing 'completed subtasks' field. "
                f"Parsed keys: {list(parsed.keys())}"
            )
        if not isinstance(completed, list):
            raise ValueError(
                f"Verifier 'completed subtasks' must be a list, "
                f"got {type(completed).__name__}"
            )

        self.last_completed = [str(s) for s in completed]
        self.last_reason = str(reason)

        # Write to context
        self.context._generated_text["llamar_completed_subtasks"] = self.last_completed
        self.context._generated_text["llamar_verifier_reason"] = self.last_reason

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
        per_robot_feedback: Optional[Dict[str, str]] = None,
        active_robot_labels: Optional[List[str]] = None,
        previous_actions: Optional[Dict[str, str]] = None,
        previous_successes: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Set all input state needed for this round at once."""
        self.known_nodes = known_nodes
        self.known_edges = known_edges
        self.open_subtasks = open_subtasks
        self.closed_subtasks = closed_subtasks
        self.memory = memory
        self.per_robot_feedback = per_robot_feedback or {}
        self.active_robot_labels = active_robot_labels or []
        self.previous_actions = previous_actions or {}
        self.previous_successes = previous_successes or {}
