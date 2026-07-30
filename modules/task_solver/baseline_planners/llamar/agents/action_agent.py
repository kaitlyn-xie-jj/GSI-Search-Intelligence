# -*- coding: utf-8 -*-
"""
LLaMAR Action Agent - Generate concrete actions for each robot

Inherits ActionNode, uses Action prompt template to build prompts,
parses LLM response into per-robot actions, subtask, reason, memory, and failure_reason.
"""
from typing import Any, Dict, List, Optional

from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.baseline_planners.llamar.prompt.action_prompt import (
    build_action_system_prompt,
    build_action_user_prompt,
    ACTION_FEW_SHOT_EXAMPLES,
)
from modules.task_solver.baseline_planners.common.prompt_components import extract_json
from modules.task_solver.baseline_planners.llamar.prompt.prompt_utils import (
    format_subtasks,
    format_feedback,
    format_previous_actions,
)
from modules.task_solver.sgi_planner.prompt.observation import format_observation


class ActionAgent(ActionNode):
    """LLaMAR Action Agent — generates concrete actions for each robot.

    Based on task instruction, observation, subtask list, memory, and feedback,
    calls LLM to generate the next action for each robot.

    Attributes:
        robot_labels: Available robot labels in the current scene.
        known_nodes: WorldModelManager node list.
        known_edges: WorldModelManager edge list.
        open_subtasks: Current pending subtask list.
        closed_subtasks: Current completed subtask list.
        memory: Accumulated scene memory string.
        current_subtask: Subtask being executed in the previous step.
        per_robot_feedback: Per-robot feedback dictionary from the previous step.
        last_actions: Most recently parsed per-robot action dictionary.
        last_subtask: Most recently parsed subtask string.
        last_reason: Most recently parsed reason string.
        last_memory: Most recently parsed memory string.
        last_failure_reason: Most recently parsed failure_reason string.
    """

    def __init__(
        self,
        logger: Logger,
        context: WorkflowContext,
        robot_labels: Optional[List[str]] = None,
        node_name: str = "LLaMAR_Action",
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
        self.active_robot_labels: List[str] = []
        self.known_nodes: List[Dict] = []
        self.known_edges: List[Dict] = []
        self.open_subtasks: Optional[List[str]] = None
        self.closed_subtasks: Optional[List[str]] = None
        self.memory: str = ""
        self.current_subtask: str = ""
        self.per_robot_feedback: Dict[str, str] = {}
        self.previous_actions: Dict[str, str] = {}
        self.previous_successes: Dict[str, bool] = {}
        
        # Few-shot config
        self.use_few_shot = use_few_shot

        # Parsed result cache
        self.last_actions: Dict[str, str] = {}
        self.last_subtask: str = ""
        self.last_reason: str = ""
        self.last_memory: str = ""
        self.last_failure_reason: str = ""

    # ------------------------------------------------------------------
    # ActionNode interface
    # ------------------------------------------------------------------

    def _build_prompt(self) -> None:
        """Build the full prompt for Action Agent.

        Includes seven input components: instruction, observation, previous actions
        (active robots only), subtasks, memory, current_subtask, feedback.
        Uses unified prompt structure: system + few-shot + user.
        """
        instruction = self.context._generated_text.get("instruction", "")
        goal_type = self.context._generated_text.get("goal_type")

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

        system_prompt = build_action_system_prompt(
            self.robot_labels,
            active_robot_labels=self.active_robot_labels,
            goal_type=goal_type,
        )
        user_prompt = build_action_user_prompt(
            instruction=instruction,
            observation=observation,
            subtasks_text=subtasks_text,
            memory=self.memory,
            current_subtask=self.current_subtask,
            feedback=feedback_text,
            previous_actions=prev_actions_text,
        )

        parts = [system_prompt]

        # Few-shot examples
        if self.use_few_shot:
            for msg in ACTION_FEW_SHOT_EXAMPLES:
                parts.append(f"\n{msg['content']}")

        parts.append(f"\n{user_prompt}")

        self.prompt = "\n".join(parts)

    async def _process_response(self, content: str) -> str:
        """Parse LLM response into per-robot actions and associated info.

        Expected JSON fields:
        - "failure reason": str
        - "memory": str
        - "reason": str
        - "subtask": str
        - "{robot_label}'s action": str  (one per robot)

        Args:
            content: Raw LLM response text.

        Returns:
            Raw response text.

        Raises:
            ValueError: On JSON parse failure, triggers ActionNode retry.
        """
        parsed = extract_json(content)

        # Extract per-robot actions (only those actually output by LLM, not all robots)
        actions: Dict[str, str] = {}
        for key, value in parsed.items():
            if key.endswith("'s action"):
                label = key.replace("'s action", "")
                if label in self.robot_labels:
                    actions[label] = str(value)

        self.last_actions = actions
        self.last_subtask = str(parsed.get("subtask", ""))
        self.last_reason = str(parsed.get("reason", ""))
        self.last_memory = str(parsed.get("memory", ""))
        self.last_failure_reason = str(parsed.get("failure reason", ""))

        # Write to context
        self.context._generated_text["llamar_actions"] = self.last_actions
        self.context._generated_text["llamar_action_subtask"] = self.last_subtask
        self.context._generated_text["llamar_action_reason"] = self.last_reason
        self.context._generated_text["llamar_action_memory"] = self.last_memory

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
        current_subtask: str = "",
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
        self.current_subtask = current_subtask
        self.per_robot_feedback = per_robot_feedback or {}
        self.active_robot_labels = active_robot_labels or []
        self.previous_actions = previous_actions or {}
        self.previous_successes = previous_successes or {}
