# -*- coding: utf-8 -*-
"""
SPINE Planning Agent - Scene-graph-based multi-turn LLM planning

Inherits ActionNode, uses SPINE prompt templates to build prompts,
achieves iterative plan-feedback loops through multi-turn dialog.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.baseline_planners.spine.prompt import (
    build_spine_system_prompt,
    build_spine_user_prompt,
    SPINE_FEW_SHOT_EXAMPLES,
)
from modules.task_solver.baseline_planners.spine.prompt.spine_observation import (
    format_spine_observation,
)
from modules.task_solver.baseline_planners.common.prompt_components import (
    extract_json,
)

logger = logging.getLogger(__name__)

# Feedback template for invalid JSON responses
INVALID_JSON_FEEDBACK = (
    "Your previous response was not valid JSON. Error: {error}\n"
    "Please respond with ONLY a valid JSON object in the specified format."
)

# Feedback template for plan validation failures
INVALID_PLAN_FEEDBACK = (
    "Your previous plan failed validation:\n{error}\n"
    "Please fix the issues and respond with a corrected plan in the specified JSON format."
)


class SPINEPlanningAgent(ActionNode):
    """SPINE planning agent -- scene-graph-based multi-turn LLM planning.

    Attributes:
        robot_labels: Available robot label list.
        known_nodes: World model node list.
        known_edges: World model edge list.
        msg_history: Multi-turn dialog history (excludes system prompt and few-shot).
        n_attempts: Max retry count within a single generate_plan call.
        base_request: Initial user instruction.
        last_request: Current round's request content (instruction for initial, update message for replan).
        last_plan: Most recent per-robot plan.
        last_reasoning: Most recent reasoning text.
        use_few_shot: Whether to use few-shot examples.
    """

    def __init__(
        self,
        logger_: Logger,
        context: WorkflowContext,
        robot_labels: Optional[List[str]] = None,
        n_attempts: int = 3,
        node_name: str = "SPINE_Planner",
        model_family: str = None,
        model_name_override: str = None,
        use_few_shot: bool = True,
    ):
        super().__init__(
            logger=logger_,
            context=context,
            next_text="",
            node_name=node_name,
            model_family=model_family,
            model_name_override=model_name_override,
        )
        self.robot_labels: List[str] = robot_labels or []
        self.known_nodes: List[Dict] = []
        self.known_edges: List[Dict] = []
        self.msg_history: List[Dict[str, str]] = []
        self.n_attempts: int = n_attempts
        self.base_request: str = ""
        self.last_request: str = ""
        self.last_plan: Optional[List[List[str]]] = None
        self.last_reasoning: str = ""
        self.use_few_shot: bool = use_few_shot

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self) -> None:
        """Build SPINE prompt (ActionNode interface compat)."""
        user_content = self._make_user_content()
        self._assemble_prompt(history=self.msg_history, user_content=user_content)

    def _make_user_content(self) -> str:
        """Build current round's user message text."""
        if not self.last_request:
            instruction = self.context._generated_text.get("instruction", "")
            self.last_request = f"task: {instruction}"
        observation = format_spine_observation(
            self.known_nodes, self.known_edges, self.robot_labels
        )
        return build_spine_user_prompt(
            request=self.last_request,
            observation=observation,
        )

    def _assemble_prompt(
        self,
        history: List[Dict[str, str]],
        user_content: str,
    ) -> None:
        """Concatenate system + few-shot + history + current user into a single prompt.

        Args:
            history: Dialog history (msg_history or temp list with retry feedback).
                     Empty for initial planning; contains user/assistant alternation for replanning.
            user_content: Current round's user message (always appears last).
        """
        goal_type = self.context._generated_text.get("goal_type")
        system_prompt = build_spine_system_prompt(
            robot_labels=self.robot_labels,
            goal_type=goal_type,
            use_few_shot=self.use_few_shot,
        )

        parts = [system_prompt]

        # Few-shot examples
        if self.use_few_shot:
            for msg in SPINE_FEW_SHOT_EXAMPLES:
                parts.append(f"\n{msg['content']}")

        # Conversation history (non-empty only during replanning)
        if history:
            for msg in history:
                parts.append(f"\n{msg['content']}")

        # Current user message
        parts.append(f"\n{user_content}")

        self.prompt = "\n".join(parts)

    async def _process_response(self, content: str) -> str:
        """Parse LLM response as JSON, extract plan field.

        plan is expected as a flat list ["robot:skill", ...] (single timestep),
        also supports nested list [["robot:skill", ...], ...] (multi-timestep).
        Flat lists are auto-wrapped as single timestep [["robot:skill", ...]].

        Args:
            content: Raw LLM response text.

        Returns:
            Raw response text (parsed results stored in self.last_plan / self.last_reasoning).

        Raises:
            ValueError: On JSON parse failure or missing required fields.
        """
        parsed = extract_json(content)

        plan = parsed.get("plan")
        if plan is None:
            raise ValueError(
                "SPINE response missing 'plan' field. "
                f"Parsed keys: {list(parsed.keys())}"
            )
        if not isinstance(plan, list):
            raise ValueError(
                f"SPINE 'plan' field must be a list, got {type(plan).__name__}"
            )

        # Detect flat list vs nested list: if first element is a string, it's flat
        if plan and isinstance(plan[0], str):
            plan = [plan]

        # Validate: list of timesteps, each is a list of "robot:skill" strings
        for ts_idx, timestep in enumerate(plan):
            if not isinstance(timestep, list):
                raise ValueError(
                    f"Timestep {ts_idx} must be a list, got {type(timestep).__name__}"
                )
            for i, entry in enumerate(timestep):
                if not isinstance(entry, str):
                    plan[ts_idx][i] = str(entry)
                entry_str = plan[ts_idx][i]
                if ":" not in entry_str:
                    raise ValueError(
                        f"Timestep {ts_idx}, entry '{entry_str}' missing ':' separator. "
                        f"Expected format: 'robot_label:skill'"
                    )

        self.last_plan = plan
        self.last_reasoning = str(parsed.get("reasoning", ""))
        return content

    # ------------------------------------------------------------------
    # Multi-turn planning with retry
    # ------------------------------------------------------------------

    async def generate_plan(
        self,
        plan_validator=None,
    ) -> Tuple[Optional[List[List[str]]], bool, List[str]]:
        """Execute planning with retry.

        msg_history management:
        - msg_history records full user/assistant alternating dialog
        - Retry feedback is only appended to a temp list, not polluting msg_history

        Prompt structure:
        - Initial planning: system + few-shot + current
        - Replanning:       system + few-shot + history + current
        - Retry:            system + few-shot + history + retry feedback + current

        Args:
            plan_validator: Optional SPINEPlanValidator instance.

        Returns:
            (plan, success, logs)
        """
        logs: List[str] = []
        user_content = self._make_user_content()
        original_user_content = user_content

        # Retry feedback appended to retry_extra, not polluting msg_history
        retry_extra: List[Dict[str, str]] = []

        for attempt in range(self.n_attempts):
            # Build prompt: history = msg_history + retry_extra
            self._assemble_prompt(
                history=self.msg_history + retry_extra,
                user_content=user_content,
            )

            try:
                response_text = await self._run_single_attempt()
            except Exception as e:
                logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
                logs.append(f"LLM call failed: {e}")
                continue

            # Try parsing JSON
            try:
                parsed = extract_json(response_text)
            except ValueError as e:
                error_msg = str(e)
                logger.warning(f"JSON parse failed on attempt {attempt + 1}: {error_msg}")
                logs.append(f"JSON parse error: {error_msg}")
                retry_extra.append({"role": "user", "content": user_content})
                retry_extra.append({"role": "assistant", "content": response_text})
                user_content = INVALID_JSON_FEEDBACK.format(error=error_msg)
                continue

            # Validate plan structure
            plan = parsed.get("plan")
            if plan is None or not isinstance(plan, list):
                error_msg = (
                    f"Missing or invalid 'plan' field. "
                    f"Got keys: {list(parsed.keys())}, "
                    f"plan type: {type(plan).__name__ if plan is not None else 'None'}. "
                    f"Expected a list of 'robot_label:skill' strings."
                )
                logger.warning(f"Plan structure invalid on attempt {attempt + 1}")
                logs.append(error_msg)
                retry_extra.append({"role": "user", "content": user_content})
                retry_extra.append({"role": "assistant", "content": response_text})
                user_content = INVALID_PLAN_FEEDBACK.format(error=error_msg)
                continue

            # Detect flat list (single timestep) vs nested list (multi-timestep)
            if plan and isinstance(plan[0], str):
                plan = [plan]

            # Normalize: ensure each timestep is a list of strings
            for ts_idx, timestep in enumerate(plan):
                if not isinstance(timestep, list):
                    plan[ts_idx] = [str(timestep)]
                else:
                    plan[ts_idx] = [str(e) for e in timestep]

            # Optional plan validator check
            if plan_validator is not None:
                is_valid, error_feedback = plan_validator.validate(plan, self.robot_labels)
                if not is_valid:
                    logger.warning(f"Plan validation failed on attempt {attempt + 1}: {error_feedback}")
                    logs.append(f"Validation error: {error_feedback}")
                    retry_extra.append({"role": "user", "content": user_content})
                    retry_extra.append({"role": "assistant", "content": response_text})
                    user_content = INVALID_PLAN_FEEDBACK.format(error=error_feedback)
                    continue

            # Success: keep the latest round's user + assistant in msg_history
            self.last_plan = plan
            self.last_reasoning = str(parsed.get("reasoning", ""))
            self.msg_history = [
                {"role": "user", "content": original_user_content},
                {"role": "assistant", "content": response_text},
            ]
            return plan, True, logs

        # All attempts exhausted
        logger.error(f"SPINE planning failed after {self.n_attempts} attempts")
        return None, False, logs

    async def _run_single_attempt(self) -> str:
        """Execute a single LLM call, bypassing ActionNode's built-in retry.

        Directly uses the underlying GPT instance to send the current prompt.

        Returns:
            LLM response text.
        """
        if self.prompt is None:
            raise ValueError("Prompt is required for SPINEPlanningAgent")

        from modules.utils.system.logging_utils import dlog
        # Structured markers for offline token counting
        self.logger.log("<!-- LLM_PROMPT_START -->", level="info", print_to_terminal=False)
        dlog(f"Prompt:\n {self.prompt}", logger=self.logger, level="debug")
        self.logger.log("<!-- LLM_PROMPT_END -->", level="info", print_to_terminal=False)

        # Access the private GPT instance
        llm = self._ActionNode__llm
        response = await llm.ask(self.prompt)

        self.logger.log("<!-- LLM_RESPONSE_START -->", level="info", print_to_terminal=False)
        dlog(f"Response:\n {response}", logger=self.logger, level="info")
        self.logger.log("<!-- LLM_RESPONSE_END -->", level="info", print_to_terminal=False)
        return response

    # ------------------------------------------------------------------
    # Public API for conversation management
    # ------------------------------------------------------------------

    def set_request(self, request: str) -> None:
        """Set the request content for the next planning round.

        Called by SPINEPlanningLayer:
        - Initial planning: set to "task: {instruction}"
        - Replanning: set to "updates: {feedback_update_message}"

        Args:
            request: Request parameter for the next user prompt.
        """
        self.last_request = request

    def reset_history(self) -> None:
        """Reset dialog history (called for new tasks)."""
        self.msg_history = []
        self.base_request = ""
        self.last_request = ""
        self.last_plan = None
        self.last_reasoning = ""

    def set_state(
        self,
        known_nodes: List[Dict],
        known_edges: List[Dict],
        robot_labels: Optional[List[str]] = None,
    ) -> None:
        """Set all observation state needed for this round at once.

        Args:
            known_nodes: World model node list.
            known_edges: World model edge list.
            robot_labels: Optional, update robot label list.
        """
        self.known_nodes = known_nodes
        self.known_edges = known_edges
        if robot_labels is not None:
            self.robot_labels = robot_labels
