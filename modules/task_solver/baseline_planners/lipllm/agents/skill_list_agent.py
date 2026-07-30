# -*- coding: utf-8 -*-
"""
LipLLM Skill List Agent - Iterative skill list generation

Inherits ActionNode, uses LipLLM prompt templates to build prompts,
iteratively calls LLM to generate the required skill list until LLM outputs "done".
Output format is robot_type:skill_str (e.g. "UAV:take_off").
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.action import ActionNode
from modules.task_solver.baseline_planners.lipllm.prompt.lipllm_prompt import (
    build_skill_list_system_prompt,
    build_skill_list_user_prompt,
)
from modules.task_solver.baseline_planners.lipllm.prompt.lipllm_examples import (
    SKILL_LIST_FEW_SHOT_EXAMPLES,
)
from modules.task_solver.baseline_planners.common.action_converter import ActionConverter
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)


def parse_typed_skill(raw: str) -> Tuple[Optional[str], str]:
    """Parse a robot_type:skill_str format skill string.

    Args:
        raw: Raw skill string, e.g. "UAV:take_off" or "take_off".

    Returns:
        (robot_type, skill_str) tuple.
        robot_type is None if no colon prefix.
    """
    if ":" in raw:
        robot_type, _, skill_str = raw.partition(":")
        return robot_type.strip(), skill_str.strip()
    return None, raw.strip()


class SkillListAgent(ActionNode):
    """Skill list agent — iteratively generates required skills until "done" is output.

    Key features:
    - Iterative generation: each LLM call outputs one robot_type:skill_str skill
    - Termination: stops when LLM outputs "done"
    - Feedback support: injects execution feedback into prompt during replanning

    Attributes:
        robot_labels: List of available robot labels.
        use_few_shot: Whether to include few-shot examples in the prompt.
        known_nodes: World model node list.
        known_edges: World model edge list.
        _current_skills: Currently generated typed skill list (robot_type:skill_str format).
        _feedback_text: Feedback text (provided by FeedbackProcessor during replanning).
        _action_converter: Action format converter for parsing non-standard skill strings.
    """

    def __init__(
        self,
        logger_: Logger,
        context: WorkflowContext,
        robot_labels: Optional[List[str]] = None,
        model_family: str = None,
        model_name_override: str = None,
        use_few_shot: bool = False,
        max_iterations: int = 15,
    ):
        super().__init__(
            logger=logger_,
            context=context,
            next_text="",
            node_name="LipLLM_SkillList",
            model_family=model_family,
            model_name_override=model_name_override,
        )
        self.robot_labels: List[str] = robot_labels or []
        self.use_few_shot: bool = use_few_shot
        self.max_iterations: int = max_iterations
        self.known_nodes: list = []
        self.known_edges: list = []
        self._current_skills: List[str] = []
        self._feedback_text: Optional[str] = None
        self._action_converter = ActionConverter(step_prefix="lipllm_step")
        self.llm_call_count: int = 0

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self) -> None:
        """Build the full prompt for SkillListAgent.

        Concatenates system prompt, few-shot examples, and user prompt into a single
        prompt string, compatible with GSI's GPT._make_request (sends only the last message).
        Uses member variables self.known_nodes / self.known_edges for environment info.
        """
        instruction = self.context._generated_text.get("instruction", "")
        goal_type = self.context._generated_text.get("goal_type")

        system_prompt = build_skill_list_system_prompt(
            robot_labels=self.robot_labels,
            goal_type=goal_type,
        )
        user_prompt = build_skill_list_user_prompt(
            instruction=instruction,
            robot_labels=self.robot_labels,
            known_nodes=self.known_nodes,
            known_edges=self.known_edges,
            current_skills=self._current_skills,
            feedback_text=self._feedback_text,
        )

        parts = [system_prompt]

        # Few-shot examples
        if self.use_few_shot:
            for example_conversation in SKILL_LIST_FEW_SHOT_EXAMPLES:
                for msg in example_conversation:
                    parts.append(f"\n{msg['content']}")

        parts.append(f"\n{user_prompt}")

        self.prompt = "\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    async def _process_response(self, response: str) -> str:
        """Parse LLM output, extract a single robot_type:skill_str skill string.

        Processing logic:
        1. Clean response text (strip whitespace, quotes, etc.)
        2. Check for "done" termination signal
        3. Extract and validate robot_type:skill_str format

        Args:
            response: Raw LLM response text.

        Returns:
            Cleaned typed skill string (robot_type:skill_str) or "done".
        """
        cleaned = response.strip().strip('"').strip("'").strip()

        # Handle multi-line response: take the first non-empty line
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if lines:
            cleaned = lines[0]

        # Strip possible numbering prefix (e.g. "1. UAV:take_off" -> "UAV:take_off")
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)

        # Strip possible quote wrapping
        cleaned = cleaned.strip('"').strip("'").strip("`").strip()

        return cleaned

    # ------------------------------------------------------------------
    # Iterative skill list generation
    # ------------------------------------------------------------------

    async def generate_skill_list(
        self,
        feedback_text: Optional[str] = None,
    ) -> List[str]:
        """Iteratively call LLM to generate the full skill list.

        Requires known_nodes / known_edges to be set via member variables before calling.

        Each iteration:
        1. Build prompt containing the current skill list
        2. Call LLM to get the next typed skill (robot_type:skill_str)
        3. Parse response, append to skill list
        4. Repeat until LLM outputs "done" or max iterations reached

        Args:
            feedback_text: Feedback text (used during replanning).

        Returns:
            List of typed skill strings (robot_type:skill_str format).
        """
        self._current_skills = []
        if feedback_text is not None:
            self._feedback_text = feedback_text

        for iteration in range(self.max_iterations):
            # Build prompt
            self._build_prompt()

            # Call LLM
            try:
                response_text = await self._run_single_attempt()
            except Exception as e:
                logger.error(
                    f"LLM call failed on iteration {iteration + 1}: {e}"
                )
                break

            # Parse response
            skill = await self._process_response(response_text)

            # Check termination
            if skill.lower() == "done":
                dlog(
                    f"SkillListAgent: generation complete after {iteration + 1} iterations, "
                    f"{len(self._current_skills)} skills generated",
                    logger=self.logger,
                    level="info",
                )
                break

            # Validate typed format and append
            if skill:
                robot_type, skill_str = parse_typed_skill(skill)
                if robot_type is None:
                    logger.warning(
                        f"Skill missing robot_type prefix: '{skill}'. "
                        f"Using original string."
                    )

                # Validate skill_str format
                parsed = self._action_converter.parse_single_action(skill_str)
                if parsed == "sync_wait" and skill_str.lower() not in ("sync_wait", "wait", "done"):
                    logger.warning(
                        f"Skill not in GSI library: '{skill_str}'. "
                        f"Using original string."
                    )

                self._current_skills.append(skill)

                dlog(
                    f"SkillListAgent: iteration {iteration + 1}, added skill: {skill}",
                    logger=self.logger,
                    level="debug",
                )
        else:
            logger.warning(
                f"SkillListAgent: reached max iterations ({self.max_iterations}), "
                f"using {len(self._current_skills)} skills collected so far"
            )

        return list(self._current_skills)

    async def _run_single_attempt(self) -> str:
        """Execute a single LLM call.

        Directly uses the underlying GPT instance to send the current prompt,
        bypassing ActionNode's built-in retry mechanism to support iterative generation.

        Returns:
            LLM response text.
        """
        if self.prompt is None:
            raise ValueError("Prompt is required for SkillListAgent")

        # Structured markers for offline token counting
        self.logger.log("<!-- LLM_PROMPT_START -->", level="info", print_to_terminal=False)
        dlog(f"Prompt:\n {self.prompt}", logger=self.logger, level="debug")
        self.logger.log("<!-- LLM_PROMPT_END -->", level="info", print_to_terminal=False)

        llm = self._ActionNode__llm
        response = await llm.ask(self.prompt)
        self.llm_call_count += 1

        self.logger.log("<!-- LLM_RESPONSE_START -->", level="info", print_to_terminal=False)
        dlog(f"Response:\n {response}", logger=self.logger, level="info")
        self.logger.log("<!-- LLM_RESPONSE_END -->", level="info", print_to_terminal=False)
        return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_feedback(self, feedback_text: str) -> None:
        """Set feedback text (used during replanning).

        Args:
            feedback_text: Feedback text formatted by FeedbackProcessor.
        """
        self._feedback_text = feedback_text

    def reset(self) -> None:
        """Reset current skill list and feedback."""
        self._current_skills = []
        self._feedback_text = None
