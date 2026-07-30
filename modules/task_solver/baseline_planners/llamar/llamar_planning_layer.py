# -*- coding: utf-8 -*-
"""
LLaMARPlanningLayer - LLaMAR planning layer implementing BaselinePlanner interface

Encapsulates the Planner -> Action -> (Verifier) three-agent collaborative loop,
driven by UnifiedTaskSolver via generate_plan and process_feedback.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from modules.task_solver.baseline_planners.base_planner import BaselinePlanner
from modules.task_solver.baseline_planners.common.action_converter import ActionConverter
from modules.task_solver.baseline_planners.llamar.agents.planner_agent import PlannerAgent
from modules.task_solver.baseline_planners.llamar.agents.action_agent import ActionAgent
from modules.task_solver.baseline_planners.llamar.agents.verifier_agent import VerifierAgent
from modules.task_solver.baseline_planners.llamar.state_manager import LLaMARStateManager
from modules.task_solver.baseline_planners.llamar.feedback_collector import FeedbackCollector
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.file import Logger
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)


class LLaMARPlanningLayer(BaselinePlanner):
    """LLaMAR planning layer — encapsulates three-agent collaborative loop.

    Implements BaselinePlanner interface, driven by UnifiedTaskSolver:
    - generate_plan: Execute one planning step (initial Planner + Action + Convert)
    - process_feedback: Collect feedback -> (Verifier) -> Planner update -> state update

    Attributes:
        max_steps: Maximum step limit.
        state_manager: LLaMAR internal state manager.
        action_converter: Action format converter.
        feedback_collector: Feedback collector.
    """

    def __init__(
        self,
        max_steps: int = 20,
        world_model_manager=None,
        context: Optional[WorkflowContext] = None,
        robot_labels: Optional[List[str]] = None,
        logger: Logger = None,
        model_family: str = None,
        model_name_override: str = None,
        use_few_shot: bool = False,
    ):
        self.max_steps = max_steps
        self.logger = logger
        self._world_model_manager = world_model_manager
        
        # Few-shot config
        self.use_few_shot = use_few_shot

        # Internal components
        self.state_manager = LLaMARStateManager()
        self.action_converter = ActionConverter()
        self.feedback_collector = FeedbackCollector()

        # Robot labels (set by factory or UnifiedTaskSolver)
        self.robot_labels: List[str] = robot_labels or []

        # Three agents (lazy init, requires context)
        self._context = context
        self._model_family = model_family
        self._model_name_override = model_name_override
        self._agents_initialized = False
        self.planner_agent: Optional[PlannerAgent] = None
        self.action_agent: Optional[ActionAgent] = None
        self.verifier_agent: Optional[VerifierAgent] = None

        # Control flags
        self._initial_plan_done = False

        # Active robot tracking
        self._active_robot_labels: List[str] = []

        # Metrics
        self._metrics: Dict[str, Any] = {
            "success": False,
            "total_steps": 0,
            "llm_calls": 0,
            "planner_calls": 0,
            "action_calls": 0,
            "verifier_calls": 0,
            "verifier_correct": None,
            "replans_full": 0,
            "replans_partial": 0,
            "planning_durations": [],
            "allocation_durations": [],
        }

        # Accumulated duration for current planning cycle (generate_plan + process_feedback merged)
        self._current_cycle_duration: float = 0.0

    def _ensure_agents(self, context: WorkflowContext) -> None:
        """Ensure agents are initialized (lazy init, created on first call)."""
        if self._agents_initialized:
            return

        ctx = context or self._context
        if ctx is None:
            raise RuntimeError("LLaMARPlanningLayer requires a WorkflowContext")

        self._context = ctx

        self.planner_agent = PlannerAgent(
            logger=self.logger,
            context=ctx,
            robot_labels=self.robot_labels,
            model_family=self._model_family,
            model_name_override=self._model_name_override,
            use_few_shot=self.use_few_shot,
        )
        self.action_agent = ActionAgent(
            logger=self.logger,
            context=ctx,
            robot_labels=self.robot_labels,
            model_family=self._model_family,
            model_name_override=self._model_name_override,
            use_few_shot=self.use_few_shot,
        )
        self.verifier_agent = VerifierAgent(
            logger=self.logger,
            context=ctx,
            robot_labels=self.robot_labels,
            model_family=self._model_family,
            model_name_override=self._model_name_override,
        )
        self._agents_initialized = True

    # =========================================================================
    # BaselinePlanner interface
    # =========================================================================

    async def generate_plan(self) -> Optional[Dict]:
        """LLaMAR one-step planning loop (with timing)."""
        _t0 = time.time()
        result = await self._generate_plan_inner()
        self._current_cycle_duration = round(time.time() - _t0, 6)
        return result

    async def _generate_plan_inner(self) -> Optional[Dict]:
        """LLaMAR one-step planning loop.

        Each call executes:
        1. If first call, invoke Planner to generate initial plan
        2. Invoke Action Agent to generate actions
        3. ActionConverter converts to dispatcher_result format
        4. If conversion fails, retry Action Agent (up to 2 times)
        5. If max retries exhausted, return None indicating planning failure
        
        Returns dispatcher_result for UnifiedTaskSolver to pass to WorldModelLayer.

        Returns:
            dispatcher_result dict (on success), or None (on planning failure).
        """
        context = self._context
        self._ensure_agents(context)
        wmm = self._get_world_model_manager(context)

        step = self.state_manager.step_num
        dlog(f"LLaMARPlanningLayer: generate_plan step={step}",
             logger=self.logger, level='info')

        # Each generate_plan call (non-first) counts as a full replan
        if step > 0:
            self._metrics["replans_full"] += 1

        # (1) Planner generates plan
        if not self._initial_plan_done:
            await self._call_planner(wmm)
            self._initial_plan_done = True

        # (2) Action Agent generates actions (with retry)
        _t_alloc = time.time()
        max_retries = 2
        dispatcher_result = None
        conversion_successful = False
        
        for attempt in range(max_retries):
            actions = await self._call_action(wmm)

            # Track active robots (those with actual actions)
            self._active_robot_labels = [
                label for label, action in actions.items()
                if action and action.lower() not in ("stay idle", "done", "wait", "sync_wait", "")
            ]

            # (3) ActionConverter converts to dispatcher_result
            dispatcher_result, needs_retry, parse_details = self.action_converter.convert(
                robot_actions=actions,
                step_num=step,
            )

            # (4) Check if retry is needed
            if not needs_retry:
                conversion_successful = True
                break
            
            if attempt == max_retries - 1:
                # Max retries exhausted, conversion still failed
                failed_info = []
                for robot, (parsed, conf, original) in parse_details.items():
                    if conf < 0.5:
                        failed_info.append(f"{robot}: '{original}' -> '{parsed}' (conf={conf:.2f})")
                
                dlog(
                    f"Action parsing failed after {max_retries} attempts:\n" + "\n".join(failed_info),
                    logger=self.logger, level='error'
                )
                self._metrics["action_parse_failures"] = self._metrics.get("action_parse_failures", 0) + 1
                self._metrics["allocation_durations"].append(round(time.time() - _t_alloc, 6))
                return None
            
            # Retry needed: log and re-invoke Action Agent
            failed_robots = [r for r, (_, conf, _) in parse_details.items() if conf < 0.5]
            dlog(
                f"Action parsing failed for {failed_robots}, retrying (attempt {attempt + 2}/{max_retries})",
                logger=self.logger, level='warning'
            )
            
            # Update feedback to hint LLM about format issues with specific error info
            feedback_hints = {}
            for robot, (parsed, conf, original) in parse_details.items():
                if conf < 0.5:
                    feedback_hints[robot] = (
                        f"IMPORTANT: Your previous action '{original}' could not be parsed correctly. "
                        f"Please use the EXACT format from the skill list. Examples:\n"
                        f"  - search<area>_for<target> (e.g., search<cybertown>_for<Blue_Truck>)\n"
                        f"  - place<object>_on<surface> (e.g., place<Box-1>_on<Table-2>)\n"
                        f"Use angle brackets < > and underscores _ exactly as shown."
                    )
            
            self.state_manager.update_feedback(feedback_hints)

        # Conversion successful, update step count and return
        if conversion_successful:
            self._metrics["allocation_durations"].append(round(time.time() - _t_alloc, 6))
            self._metrics["total_steps"] = step + 1
            return dispatcher_result
        
        # Should not reach here, but as a safety fallback
        return None

    async def process_feedback(
        self,
        exec_result: Dict[str, Any],
        newcase_events: List[Dict],
        context,
    ) -> None:
        """Process execution feedback (with timing), optionally call Verifier, then Planner update.

        Args:
            exec_result: Execution result dictionary containing outcomes.
            newcase_events: List of unexpected events pushed by EventBus.
            context: WorkflowContext instance.
        """
        _t0 = time.time()
        await self._process_feedback_inner(exec_result, newcase_events, context)
        cycle_total = round(self._current_cycle_duration + (time.time() - _t0), 6)
        self._metrics["planning_durations"].append(cycle_total)
        self._current_cycle_duration = 0.0

    async def _process_feedback_inner(
        self,
        exec_result: Dict[str, Any],
        newcase_events: List[Dict],
        context,
    ) -> None:
        """Actual logic for process_feedback."""
        outcomes = exec_result.get('outcomes', []) or []
        self._ensure_agents(context)
        wmm = self._get_world_model_manager(context)

        # (a) Collect feedback
        for evt in newcase_events:
            self.feedback_collector._pending_newcase_events.append(evt)

        feedback = self.feedback_collector.collect_feedback(
            outcomes=outcomes,
            robot_labels=self.robot_labels,
            active_robot_labels=self._active_robot_labels,
        )
        self.state_manager.update_feedback(feedback)

        # (b) Record action execution results
        successes = self._extract_successes(outcomes)
        self.state_manager.record_actions(
            self.action_agent.last_actions if self.action_agent else {},
            successes,
        )

        # (c) Verifier
        completed = await self._call_verifier(wmm, feedback)
        self.state_manager.update_completed(completed)

        # (d) Planner Agent updates plan
        await self._call_planner(wmm)

        # (e) Increment step
        self.state_manager.increment_step()

    def is_task_completed(self) -> bool:
        """Determine whether the task is completed.

        Returns True when all open_subtasks are empty (all completed).
        """
        open_subs = self.state_manager.open_subtasks
        if open_subs is None:
            return False
        return len(open_subs) == 0

    def reset(self) -> None:
        """Reset internal state."""
        self.state_manager.reset()
        self._initial_plan_done = False
        self._active_robot_labels = []
        self._current_cycle_duration = 0.0
        self._metrics = {
            "success": False,
            "total_steps": 0,
            "llm_calls": 0,
            "planner_calls": 0,
            "action_calls": 0,
            "verifier_calls": 0,
            "verifier_correct": None,
            "replans_full": 0,
            "replans_partial": 0,
            "planning_durations": [],
            "allocation_durations": [],
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get experiment metrics."""
        return dict(self._metrics)

    # =========================================================================
    # Agent call helpers
    # =========================================================================

    async def _call_planner(self, wmm) -> None:
        """Call Planner Agent and update state."""
        self.planner_agent.set_state(
            known_nodes=wmm.known_nodes,
            known_edges=wmm.known_edges,
            open_subtasks=self.state_manager.open_subtasks,
            closed_subtasks=self.state_manager.closed_subtasks,
            memory=self.state_manager.memory,
        )
        await self.planner_agent.run(auto_next=False)
        self._metrics["planner_calls"] += 1
        self._metrics["llm_calls"] += 1

        if self.planner_agent.last_plan is not None:
            self.state_manager.update_plan(self.planner_agent.last_plan)
        dlog(f"LLaMAR Planner: {self.planner_agent.last_reason}",
             logger=self.logger, level='info')

    async def _call_action(self, wmm) -> tuple:
        """Call Action Agent, return actions."""
        self.action_agent.set_state(
            known_nodes=wmm.known_nodes,
            known_edges=wmm.known_edges,
            open_subtasks=self.state_manager.open_subtasks,
            closed_subtasks=self.state_manager.closed_subtasks,
            memory=self.state_manager.memory,
            current_subtask=self.state_manager.current_subtask,
            per_robot_feedback=self.state_manager.previous_feedback,
            active_robot_labels=self._active_robot_labels,
            previous_actions=self.state_manager.previous_actions,
            previous_successes=self.state_manager.previous_successes,
        )
        await self.action_agent.run(auto_next=False)
        self._metrics["action_calls"] += 1
        self._metrics["llm_calls"] += 1

        actions = self.action_agent.last_actions
        memory = self.action_agent.last_memory
        subtask = self.action_agent.last_subtask
        self.state_manager.update_memory(memory)
        self.state_manager.update_current_subtask(subtask)

        dlog(f"LLaMAR Action: subtask={subtask}, reason={self.action_agent.last_reason}",
             logger=self.logger, level='info')
        return actions

    async def _call_verifier(self, wmm, feedback: Dict[str, str]) -> List[str]:
        """Call Verifier Agent, return list of completed subtasks."""
        self.verifier_agent.set_state(
            known_nodes=wmm.known_nodes,
            known_edges=wmm.known_edges,
            open_subtasks=self.state_manager.open_subtasks,
            closed_subtasks=self.state_manager.closed_subtasks,
            memory=self.state_manager.memory,
            per_robot_feedback=feedback,
            active_robot_labels=self._active_robot_labels,
            previous_actions=self.state_manager.previous_actions,
            previous_successes=self.state_manager.previous_successes,
        )
        await self.verifier_agent.run(auto_next=False)
        self._metrics["verifier_calls"] += 1
        self._metrics["llm_calls"] += 1

        completed = self.verifier_agent.last_completed
        
        dlog(f"LLaMAR Verifier: completed={completed}, reason={self.verifier_agent.last_reason}",
             logger=self.logger, level='info')
        return completed

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _get_world_model_manager(self, context):
        """Get WorldModelManager instance.

        Prefers externally injected reference, falls back to context attribute.
        """
        if self._world_model_manager is not None:
            return self._world_model_manager
        return getattr(context, '_world_model_manager', None)

    def set_world_model_manager(self, wmm) -> None:
        """Set WorldModelManager reference (called by UnifiedTaskSolver)."""
        self._world_model_manager = wmm

    def set_robot_labels(self, labels: List[str]) -> None:
        """Set robot label list (called by factory or UnifiedTaskSolver)."""
        self.robot_labels = labels
        # Update already-initialized agents
        if self.planner_agent:
            self.planner_agent.robot_labels = labels
        if self.action_agent:
            self.action_agent.robot_labels = labels
        if self.verifier_agent:
            self.verifier_agent.robot_labels = labels

    @staticmethod
    def _extract_successes(outcomes: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Extract per-robot success status from outcomes."""
        successes: Dict[str, bool] = {}
        for outcome in outcomes:
            meta = outcome.get("meta") or {}
            robot = meta.get("robot_label", "")
            success = meta.get("success")
            if success is None:
                success = (outcome.get("data") or {}).get("success", False)
            if robot:
                successes[robot] = bool(success)
        return successes
