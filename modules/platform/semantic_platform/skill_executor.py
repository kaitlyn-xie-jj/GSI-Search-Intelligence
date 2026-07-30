import asyncio
import random
import copy
import contextlib
import time
import logging
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm

from modules.config.base.enums import SkillName
from modules.config.entities.skill_config import skill_template_manager, extract_runtime_watch_specs
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.semantic_platform.scene_graph_manager import SemanticSceneGraph
from modules.platform.semantic_platform.context_hub import ContextHub
from modules.platform.semantic_platform.new_case_generator import NewCaseGenerator
from modules.platform.semantic_platform.new_case_injector import NewCaseInjector
from modules.platform.semantic_platform.skill_outcome_postprocessor import OutcomePostProcessor
from modules.platform.semantic_platform.skill_feedback_generator import FeedbackGenerator
from modules.platform.semantic_platform.runtime_guards import GuardRegistry, RuntimeGuardMonitor
from modules.platform.semantic_platform.new_case_controller import NewCaseController
from modules.platform.semantic_platform.skill_parameter_processor import SkillParameterProcessor
from modules.events.event_bus import publish_event_sync
from modules.config.events import NewCaseEvent
from modules.utils.system.var_dump import dump_var
from modules.utils.system.logging_utils import dlog
from modules.config.system_config import config

logger = logging.getLogger(__name__)


class SkillExecutionInterrupted(Exception):
    """Skill execution interruption exception."""
    def __init__(self, reason: str, event: Any = None):
        super().__init__(reason)
        self.reason = reason
        self.event = event


class SkillExecutor:
    """
    Execution layer responsible for skill execution, motion simulation, and environment interaction.
    """
    
    def __init__(self, scene_graph: AbstractSceneGraph,
                 enable_visualization: bool = True,
                 fine_grained_simulation: bool = True,
                 enable_video_recording: bool = False,
                 video_output_path: Optional[str] = None,
                 logger = None,
        is_replay: bool = False):
        """
        Initialize the skill executor.
        """
        self.scene_graph = scene_graph
        self.logger = logger or logging.getLogger(__name__)
        self.knowledge_scope = 'local'
        self.is_replay = is_replay
        self._replay_meta: Optional[Dict[str, Any]] = None
        self.enable_visualization = config.enable_visualization if config.enable_visualization is not None else enable_visualization
        self.enable_video_recording = config.enable_video_recording if config.enable_video_recording is not None else enable_video_recording
        self.fine_grained_simulation = config.fine_grained_simulation if config.fine_grained_simulation else fine_grained_simulation
        if self.enable_visualization and not self.fine_grained_simulation:
            self.fine_grained_simulation = True
        self.new_case_mode = config.new_case_mode or "immediate"
        
        # Mapping tables
        self.label_to_id_map = scene_graph.get_node_map(map_type='label_to_id') if scene_graph else {}
        self.id_to_label_map = scene_graph.get_node_map(map_type='id_to_label') if scene_graph else {}
        
        # Context hub, set by the world model layer
        self.context_hub: Optional[ContextHub] = None
        
        # Interrupt control (global stop signal)
        self._interrupt_event = asyncio.Event()
        self._interrupt_payload: Any = None
        
        # Current execution context (robot_label -> execution info)
        self._current_executions: Dict[str, Dict] = {}
        
        # Visualization components
        self.visualizer = None
        self.video_writer = None
        
        if self.enable_visualization:
            self._init_visualization(self.enable_video_recording, video_output_path)
        
        # Motion controller
        self._init_motion_controller()

        self.guard_registry = GuardRegistry(self.scene_graph)
        self.guard_monitor = RuntimeGuardMonitor(self.guard_registry, interval=0.25)

        # Skill parameter processor
        self.param_processor: Optional[SkillParameterProcessor] = None
        if scene_graph:
            self.param_processor = SkillParameterProcessor(
                scene_graph, self.label_to_id_map, self.id_to_label_map
            )

        # New-case generation/injection services injected by the upper layer
        self.newcase_generator: Optional[NewCaseGenerator] = None
        self.newcase_injector: Optional[NewCaseInjector] = None
        self.newcase_ctrl: Optional[NewCaseController] = None

        self._incremental_outcome_handler = None

        # Injection policy
        self.runtime_injection_prob  = 0.0
        self.runtime_injection_interval = 0.0
        self._last_runtime_inject_ts = 0.0
    
    def _init_visualization(self, enable_video_recording: bool, video_output_path: Optional[str]):
        """Initialize visualization components."""
        from .visualization import RealTimeScenarioVisualizer
        self.visualizer = RealTimeScenarioVisualizer(
            title="Multi-Robot Task Execution",
            enable_motion_trails=True,
            motion_update_rate=20,
            goal=self.scene_graph._goal if self.scene_graph and self.scene_graph._goal else None
        )
        self.visualizer.fig.canvas.draw()
        plt.pause(0.05)
        
        if enable_video_recording:
            from .visualization import VideoWriter
            self.video_writer = VideoWriter(
                self.visualizer.fig,
                video_output_path,
                fps=12
            )
            self.visualizer.set_video_writer(self.video_writer)
    
    def _init_motion_controller(self):
        """Initialize the motion controller."""
        motion_simulator = None
        if self.fine_grained_simulation:
            from modules.platform.semantic_platform.skill_simulator_enhance import EnhancedSkillMotionSimulator
            # Decide whether to use fast mode based on config
            fast_mode = not config.simulate_time_delay
            motion_simulator = EnhancedSkillMotionSimulator(
                update_interval=0.05,
                fast_mode=fast_mode
            )
        
        from modules.platform.semantic_platform.motion_controller import MotionController
        self.motion_controller = MotionController(
            motion_simulator=motion_simulator,
            visualizer=self.visualizer,
            scene_graph=self.scene_graph,
            enable_visualization=self.enable_visualization
        )
    
    def set_context_hub(self, context_hub: ContextHub):
        """Set the context hub."""
        self.context_hub = context_hub
        # Sync motion controller to the context hub
        if self.context_hub:
            self.context_hub.set_motion_controller(self.motion_controller)
    
    def set_new_case_services(self, generator: NewCaseGenerator, injector: NewCaseInjector, ctrl: NewCaseController):
        """Inject the new-case generator and injector."""
        # Set new-case services only when new-case generation is enabled
        if config.enable_new_case_generation:
            self.newcase_generator = generator
            self.newcase_injector = injector
            self.newcase_ctrl = ctrl
            if hasattr(self.newcase_generator, "set_op_usage_resolver") and hasattr(self.newcase_ctrl, "get_op_usage_count"):
                self.newcase_generator.set_op_usage_resolver(self.newcase_ctrl.get_op_usage_count)
        else:
            self.newcase_generator = None
            self.newcase_injector = None
            self.newcase_ctrl = None

    def set_replay_meta(self, meta: Optional[Dict[str, Any]]):
        self._replay_meta = meta
        if self.newcase_generator is not None and hasattr(self.newcase_generator, "set_replay_meta"):
            self.newcase_generator.set_replay_meta(meta)
        if self.newcase_ctrl is not None and hasattr(self.newcase_ctrl, "set_replay_meta"):
            self.newcase_ctrl.set_replay_meta(meta)
    
    def set_knowledge_scope(self, scope: str):
        if scope not in ['local', 'global']:
            logger.warning(f"Invalid knowledge scope: {scope}. Using 'local' as default.")
            scope = 'local'
        self.knowledge_scope = scope

    def set_incremental_outcome_handler(self, handler):
        self._incremental_outcome_handler = handler

    # --------------------------- NewCaseEvent Publishing ---------------------------

    async def _publish_new_case_event(self, case_id: str, description: Any, entity_id: str,
                                      entity_type: str, priority: int, context: Dict[str, Any]) -> None:
        new_case_event = NewCaseEvent(
            case_id=case_id, 
            description=description,
            entity_id=str(entity_id or ""),
            entity_type=str(entity_type or ""),
            timestamp=datetime.now()
        )
        await publish_event_sync(new_case_event)
        dlog(f"newcase_event: {description}", logger=self.logger, level='warning')

    # --------------------------- Main Flow ---------------------------

    async def execute_plan(self, skills_by_timestep: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a complete skill plan.
        Returns:
            Execution result. If stopped midway due to a new case, success=False.
        """
        # Reset state
        self._interrupt_event.clear()
        self._interrupt_payload = None
        
        # Start context services
        if self.context_hub:
            await self.context_hub.start()
        
        all_outcomes: List[Dict] = []
        execution_success = True

        # Plan level: reset and select the unique instance
        if self.newcase_ctrl:
            self.newcase_ctrl.reset_for_new_plan()
            if self.newcase_ctrl.should_attempt_injection_this_plan():
                # Let the controller select one (timestep, robot_id, skill) instance from this plan
                self.newcase_ctrl.select_skill_instance(skills_by_timestep)
        
        try:
            # Sort timesteps and execute
            for timestep in sorted(skills_by_timestep.keys()):
                # If any skill has already triggered a stop, stop here directly
                if self._interrupt_event.is_set():
                    logger.warning(f"Execution interrupted before timestep {timestep}")
                    execution_success = False
                    break
                
                logger.info(f"--- Executing timestep {timestep} ---")
                timestep_skills = skills_by_timestep[timestep]

                # Execute all skills at the current timestep concurrently, one task per robot
                tasks = [asyncio.create_task(self._execute_one(robot_label, skill_info, timestep))
                        for robot_label, skill_info in timestep_skills.items()]

                # Wait in a loop; once any task completes and global interrupt is set, cancel remaining tasks
                pending = set(tasks)
                any_failed = False
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    # Collect completed results
                    for d in done:
                        try:
                            res = d.result()
                            if not res.get("success", False):
                                any_failed = True
                        except Exception as e:
                            any_failed = True
                            dlog(f"Task exception: {e}", logger=self.logger, level="error")

                    # If any task causes a global interrupt by publishing a new-case event, cancel remaining tasks
                    if self._interrupt_event.is_set():
                        for p in pending:
                            p.cancel()
                        # Wait for cancellation to complete, swallowing cancellation exceptions
                        await asyncio.gather(*pending, return_exceptions=True)
                        pending.clear()
                        any_failed = True
                        break  # Exit timestep loop

                if any_failed:
                    execution_success = False

                # If interrupted, end quickly and avoid draining large queues to keep the system responsive
                if self._interrupt_event.is_set():
                    logger.warning(f"Execution interrupted at timestep {timestep}")
                    break

                if self.context_hub:
                    # Wait for context quiescence
                    dlog(f"Awaiting quiescence after timestep {timestep}...", logger=self.logger, level='info')
                    quiet = await self.context_hub.await_quiescence(timeout=1.0, include_state=True)
                    if not quiet:
                        logger.warning(f"ContextHub did not become quiet within the timeout after timestep {timestep}.")

                    # Collect outcomes for this timestep
                    step_outcomes = await self.context_hub.collect_outcomes(drain=True)
                    if step_outcomes:
                        all_outcomes.extend(step_outcomes)

                        # Incrementally report to the upper layer (world model / planning layer)
                        if self._incremental_outcome_handler is not None:
                            try:
                                await self._incremental_outcome_handler(step_outcomes)
                            except Exception as e:
                                logger.error(f"Incremental outcome handler error: {e}", exc_info=True)
                
                # Update visualization
                if self.enable_visualization and self.visualizer:
                    self.visualizer.update(
                        new_nodes=self.scene_graph._nodes,
                        new_edges=self.scene_graph._edges,
                        timestep=timestep
                    )
                
                if self.video_writer:
                    self.video_writer.update()
        
        finally:
            # Stop context services
            if self.context_hub:
                try:
                    await self.context_hub.await_quiescence(timeout=1.0, include_state=True)
                except Exception:
                    pass
                await self.context_hub.stop()
        
        # Merge outcomes
        merged_outcomes = OutcomePostProcessor.merge_outcomes(all_outcomes)
        
        return {
            'success': execution_success,
            'outcomes': merged_outcomes
        }
    
    async def _execute_one(self, robot_label: str, skill_info: Dict[str, Any], timestep: int) -> Dict[str, Any]:
        """Execute one skill for one robot."""
        skill = skill_info.get('skill')
        params = skill_info.get('params', {})
        
        # Sync wait passes directly
        if skill == 'sync_wait':
            logger.info(f"Robot '{robot_label}' performing sync wait at timestep {timestep}")
            return {'robot_label': robot_label, 'skill': skill, 'success': True}
        
        # Get robot ID
        robot_id = self.label_to_id_map.get(robot_label, robot_label)
        robot = self.scene_graph.get_robot(robot_id)
        if not robot:
            logger.error(f"Robot not found: {robot_label}")
            return {'robot_label': robot_label, 'skill': skill, 'success': False}
        
        # Parameter processing happens at execution time, not during plan translation
        if self.param_processor:
            params = self.param_processor.process_skill_parameters(
                skill=skill, params=params, robot_id=robot_id
            )
        
        # Get skill execution information
        exec_info = skill_template_manager.get_execution_info(skill)
        base_time = exec_info['base_time']
        base_energy = exec_info['base_energy']
        
        # Calculate execution time and energy use
        if self.param_processor:
            exec_time, energy_used = self.param_processor.calculate_execution_time_and_energy(
                skill=skill, params=params, base_time=base_time, base_energy=base_energy
            )
        else:
            exec_time = params.get('execution_time', base_time)
            energy_used = params.get('energy_used', base_energy)
        
        # Update execution time and energy use in params
        params['execution_time'] = exec_time
        params['energy_used'] = energy_used
        
        # Record execution context
        self._start_skill_execution(robot_label, skill_info, timestep)
        context = self._build_skill_context(robot, skill, params)
        post_exec_event: Optional[Tuple[str, Dict]] = None  # Stores info events to publish

        # Dynamic check gating: enabled only for the selected instance when quota remains
        check_dynamic = skill_template_manager.get_dynamic_check_setting(skill)
        if self.newcase_ctrl:
            if self.newcase_ctrl.disable_all_dynamic_checks():
                check_dynamic = False
            else:
                # Both manual template setting and automatic setting must pass
                check_dynamic = check_dynamic and self.newcase_ctrl.dynamic_check_enabled_for_instance(
                    skill_name=skill, timestep=timestep, robot_label=robot_label
                )

        try:
            # Selected unique instance: always try one injection before precheck.
            # No injection occurs if quota is exhausted or this instance is not selected.
            if check_dynamic and self.newcase_ctrl and self.newcase_ctrl.should_attempt_injection_this_plan():
                injected_event = self.newcase_generator.try_generate_once(
                    robot_id=str(robot_id), skill=skill, params=params, context=context
                )
                if injected_event:
                    await self.newcase_injector.inject_event_now(injected_event)
                    # Counts: skill + new-case type
                    self.newcase_ctrl.record_generated_for_skill(skill)
                    self.newcase_ctrl.record_generated_for_event(injected_event)

            # Precondition check
            precond_status, outcome_code, details = await self._check_skill_precondition(
                robot, skill, params, check_dynamic_conditions=True
            )
            
            # Precheck failed: publish new-case event and stop globally
            if precond_status == "fail":
                events_to_publish: List[Dict[str, Any]] = []
                if self.new_case_mode == "aggregate" and isinstance(details, dict) and "all_events" in details:
                    for ev in details["all_events"]:
                        if ev.get("kind") != "fail":
                            continue
                        events_to_publish.append(ev)
                else:
                    events_to_publish.append({
                        "kind": "fail",
                        "event_key": outcome_code,
                        "event": details or {},
                        "rule": None,
                    })

                # Publish NewCaseEvent one by one
                for idx, ev in enumerate(events_to_publish):
                    ev_key = ev.get("event_key") or (outcome_code or "precheck_failed")
                    ev_payload = ev.get("event") or {}
                    description = {
                        "phase": "precheck",
                        "skill": skill,
                        "task_id": context.get("task_id"),
                        "reason": ev_key,
                        "event_kind": "incident",
                        "details": ev_payload,
                        "robot": {
                            "id": context.get("robot_id"),
                            "label": context.get("robot_label"),
                            "type": context.get("robot_type"),
                            "location": context.get("robot_location"),
                        },
                        "object": {
                            "id": context.get("object_id"),
                            "location": context.get("target_location"),
                        } if context.get("object_id") else None
                    }
                    case_id = f"{ev_key}-{context.get('robot_id')}-{idx}"
                    await self._publish_new_case_event(
                        case_id=case_id,
                        description=description,
                        entity_id=str(context.get("robot_id")),
                        entity_type="robot",
                        priority=1,
                        context=context
                    )
                # Overall result: this skill fails and global stop is triggered
                self._interrupt_execution({"phase": "precheck", "reason": outcome_code})
                return {
                    'robot_label': robot_label,
                    'skill': skill,
                    "task_id": context.get("task_id"),
                    'exec_time': 0.0,
                    'energy_used': 0.0,
                    'success': False,
                    'outcome': outcome_code or "precheck_failed",
                    'details': details or {}
                }

            post_exec_event = None
            if precond_status == "info":
                if self.new_case_mode == "aggregate" and isinstance(details, dict) and "all_events" in details:
                    infos = [ev for ev in details["all_events"] if ev.get("kind") == "info"]
                    post_exec_event = infos  # list[dict]
                else:
                    post_exec_event = [{
                        "kind": "info",
                        "event_key": outcome_code,
                        "event": details or {},
                        "rule": None,
                    }]

            # Precheck passed; start execution
            try:
                simulation_task = asyncio.create_task(
                    self._simulate_skill(robot_id, skill, params, skill_info, context, check_dynamic)
                )
                await self._wait_for_completion_or_interrupt(exec_time)
                result = await simulation_task
                dlog(f"Skill completed: {robot_label} executed {skill}")
                
            except SkillExecutionInterrupted as e:
                # Runtime interruption
                logger.warning(f"Skill execution interrupted: {e.reason}")
                result = {
                    'success': False,
                    'outcome': 'interrupted',
                    'details': {'reason': e.reason}
                }
            
            # Apply skill effects
            await self._apply_skill_effects(robot, skill, params, result)
            
            # If precheck requests publishing info events and skill execution succeeded
            if post_exec_event and result.get('success', True):
                for idx, ev in enumerate(post_exec_event):
                    info_code = ev.get("event_key")
                    info_details = ev.get("event") or {}
                    description = {
                        "phase": "postexec",
                        "skill": skill,
                        "task_id": context.get("task_id"),
                        "reason": info_code,
                        "event_kind": "info",
                        "details": info_details,
                        "robot": {
                            "id": context.get("robot_id"),
                            "label": context.get("robot_label"),
                            "type": context.get("robot_type"),
                            "location": context.get("robot_location"),
                        },
                        "object": {
                            "id": context.get("object_id"),
                            "location": context.get("target_location"),
                        } if context.get("object_id") else None
                    }
                    await self._publish_new_case_event(
                        case_id=str(info_code or f"info-{skill}-{robot_id}-{idx}"),
                        description=description,
                        entity_id=str(context.get("robot_id")),
                        entity_type="robot",
                        priority=3,
                        context=context
                    )

            return {
                'robot_label': robot_label,
                'skill': skill,
                "task_id": context.get("task_id"),
                'exec_time': exec_time,
                'energy_used': energy_used,
                **result
            }
        
        finally:
            self._end_skill_execution(robot_label)
    
    async def _simulate_skill(self, robot_id: int, skill: str, params: Dict,
                              skill_info: Dict, context: Dict, check_dynamic_conditions: bool) -> Dict:
        """
        Simulate skill execution, runtime perception, and runtime injection.
        If a guard fails, publish a new-case event immediately and stop globally.
        """
        # Build context
        watch_specs = []
        if check_dynamic_conditions:
            watch_specs = extract_runtime_watch_specs(skill, params, context)

        async def _motion():
            if self.fine_grained_simulation:
                res = await self.motion_controller.simulate_skill_motion(
                    robot_id=robot_id,
                    skill=skill,
                    params=params,
                    interrupt_check=lambda: self._interrupt_event.is_set()
                )
                # Clean up display elements
                if self.enable_visualization:
                    asyncio.create_task(self.motion_controller.cleanup_skill_display(robot_id))
                    if skill == SkillName.TAKE_PHOTO.value:
                        asyncio.create_task(self.motion_controller.cleanup_task_labels(skill, params.get('object_id')))
                return res
            else:
                return {'success': True}

        async def _runtime_watch_and_inject():
            if not watch_specs and (not check_dynamic_conditions or self.runtime_injection_prob <= 0.0):
                return None

            interval = min(self.guard_monitor.interval, self.runtime_injection_interval)
            while True:
                if self._interrupt_event.is_set():
                    return None  # External interrupt

                # 1) Guard monitoring
                if watch_specs:
                    failure = await self.guard_monitor.watch(
                        guards=watch_specs,
                        context=context,
                        params=params,
                        interrupt_flag_fn=lambda: self._interrupt_event.is_set()
                    )
                    if failure:
                        # Runtime guard failed: publish new-case event immediately and stop globally
                        description = {
                            "phase": "runtime",
                            "skill": skill,
                            "reason": failure.get("on_fail") or failure.get("reason") or "guard_failed",
                            "event_kind": "incident",
                            "details": {
                                "guard": failure.get("guard_name"),
                                "severity": failure.get("severity"),
                                "message": failure.get("message"),
                                "raw": failure
                            },
                            "robot": {
                                "id": context.get("robot_id"),
                                "label": context.get("robot_label"),
                                "type": context.get("robot_type"),
                                "location": context.get("robot_location"),
                            },
                            "object": {
                                "id": context.get("object_id"),
                                "location": context.get("target_location"),
                            } if context.get("object_id") else None
                        }
                        case_id = f"runtime-{skill}-{context.get('robot_id')}"
                        await self._publish_new_case_event(
                            case_id=case_id,
                            description=description,
                            entity_id=str(context.get("robot_id")),
                            entity_type="robot",
                            priority=2,
                            context=context
                        )
                        # Global stop
                        self._interrupt_execution({"phase": "runtime", "reason": description["reason"]})
                        return failure  # Notify upper layer of failure

                # 2) Probabilistic runtime injection (non-blocking)
                if check_dynamic_conditions:
                    now = time.time()
                    if (self.newcase_generator and self.newcase_injector
                        and self.runtime_injection_prob > 0.0
                        and self.newcase_generator.can_generate(now)
                        and random.random() < self.runtime_injection_prob
                        and (now - self._last_runtime_inject_ts) >= self.runtime_injection_interval):
                        evt = self.newcase_generator.generate_event(skill_info, context)
                        if evt:
                            asyncio.create_task(self.newcase_injector.inject_event_now(evt))
                            self._last_runtime_inject_ts = now

                await asyncio.sleep(interval)

        # No guards or no injection: do not create a watch task; run motion directly to avoid concurrent races
        if not watch_specs or (not check_dynamic_conditions or self.runtime_injection_prob <= 0.0):
            return await _motion()

        # Concurrent execution: motion + monitoring/injection
        motion_task = asyncio.create_task(_motion())
        watch_task  = asyncio.create_task(_runtime_watch_and_inject())

        done, _ = await asyncio.wait({motion_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)

        # Monitor finishes first
        if watch_task in done:
            failure = await watch_task  # May be None or a failure dict
            if failure:
                # Monitor reports failure: cancel motion and return failure result
                motion_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await motion_task
                return {
                    "success": False,
                    "outcome": failure.get("on_fail") or failure.get("reason") or "guard_failed",
                    "details": {
                        "guard": failure.get("guard_name"),
                        "severity": failure.get("severity"),
                        "message": failure.get("message"),
                        "reason": failure.get("reason"),
                    }
                }
            # Monitor ended early without failure, for example due to external interrupt; wait for motion to finish
            res = await motion_task
            return res

        # Motion finishes first: cancel monitor task
        if not watch_task.done():
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch_task

        return await motion_task
    
    async def _wait_for_completion_or_interrupt(self, exec_time: float):
        """Wait for skill completion or interruption."""
        # Decide whether to simulate time delay based on time simulation config
        if config.simulate_time_delay:
            interval = 0.05
            elapsed = 0.0
            half_width = shutil.get_terminal_size().columns // 2
            progress_format = "Skill Progress: |{bar}| {n:.1f}s / {total:.1f}s"
            
            with tqdm(total=exec_time, bar_format=progress_format, ncols=half_width) as pbar:
                while elapsed < exec_time:
                    if self._interrupt_event.is_set():
                        raise SkillExecutionInterrupted(
                            reason="External interruption",
                            event=self._interrupt_payload
                        )
                    await asyncio.sleep(interval)
                    increment = min(interval, exec_time - elapsed)
                    elapsed += increment
                    pbar.update(increment)
                    
                    if self.enable_visualization and self.visualizer:
                        if int(elapsed * 10) % 2 == 0:
                            plt.pause(0.001)
        else:
            # Complete quickly when time simulation is not needed
            if self._interrupt_event.is_set():
                raise SkillExecutionInterrupted(
                    reason="External interruption",
                    event=self._interrupt_payload
                )
            # Wait briefly to preserve async behavior
            await asyncio.sleep(0.001)
    
    async def _check_skill_precondition(self, robot: Dict, skill: str, params: Dict, check_dynamic_conditions: bool = True) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Unified precheck."""
        context = self._build_skill_context(robot, skill, params)
        status, reason, event = skill_template_manager.evaluate_precheck(
            robot, skill, params, context, 
            check_dynamic_conditions=check_dynamic_conditions
        )
        return status, reason, (event or {})
    
    async def _apply_skill_effects(self, robot: Dict, skill: str,
                                  params: Dict, result: Dict) -> None:
        """Apply skill effects."""
        success = result.get('success', True)
        outcome = 'success' if success else 'failure'
        context = self._build_skill_context(robot, skill, params)
        
        # Get operations for the environment graph
        operations = skill_template_manager.apply_effects(robot, skill, outcome, params, context)
        
        # Send operations to the context hub
        if operations and self.context_hub:
            await self.context_hub.enqueue_ops(operations, context)
        
        # Get skill feedback outcomes
        outcomes = self._get_skill_outcomes(skill, outcome, params, context)
        outcomes = FeedbackGenerator.merge_into_outcomes(
            skill=skill,
            success=success,
            params=params,
            context=context,
            skill_result=result,
            skill_info={"skill": skill, "params": params},
            outcomes=outcomes
        )
        
        # Send outcomes to the context hub
        if outcomes and self.context_hub:
            meta = {
                "robot_id": context.get("robot_id"),
                "robot_label": context.get("robot_label"),
                "skill": skill,
                "task_id": context.get("task_id"),
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }
            await self.context_hub.enqueue_outcomes(outcomes, meta)
    
    def _build_skill_context(self, robot: Dict, skill: str, params: Dict) -> Dict:
        """Build skill execution context."""
        context = {
            "robot": robot,
            "robot_id": robot['id'],
            "robot_label": robot.get('properties', {}).get('label'),
            "robot_type": robot.get('properties', {}).get('type'),
            "graph": self.scene_graph,
            "task_id": params.get("task_id"),
            "label_to_id_map": self.label_to_id_map,
            "skill": skill,
        }
        
        # Robot location
        context["robot_location"] = self._get_node_location(robot)
        context["robot_location_label"] = self._get_node_location(robot, return_label=True)
        
        # Target object
        if params.get('object_id'):
            primary_target_id = params.get('object_id')
            obj = self.scene_graph.get_node_by_id(primary_target_id)
            if obj:
                context["object_id"] = primary_target_id
                context["target_node"] = obj
                context["target_label"] = obj.get("properties", {}).get("label")
                context["target_location"] = self._get_node_location(obj)
                context["target_location_label"] = self._get_node_location(obj, return_label=True)
        
        # Destination
        context["dest"] = params.get('dest')
        if params.get('destination_id'):
            destination_id = params.get('destination_id')
            destination = self.scene_graph.get_node_by_id(destination_id)
            if destination:
                context["destination_id"] = destination_id
                context["destination_node"] = destination
                context["destination_label"] = destination.get("properties", {}).get("label")

        # Identify and handle secondary target objects
        if params.get("surface_id"):
            surface_id = params["surface_id"]
            surface = self.scene_graph.get_node_by_id(surface_id)
            if surface:
                context["surface_id"] = surface_id
                context["surface_node"] = surface
                context["surface_subtype"] = surface.get("properties", {}).get("subtype")
                context["surface_label"] = surface.get("properties", {}).get("label")
                context["surface_location"] = self._get_node_location(surface)
                context["surface_location_label"] = self._get_node_location(surface, return_label=True)
        
        # Carrying relation
        carried_ids = self.scene_graph.get_neighbors_by_relation(robot['id'], 'carrying')
        if carried_ids:
            context["carried_object_ids"] = carried_ids

        # Carrier
        if params.get("carrier_id"):
            carrier_id = params["carrier_id"]
            carrier = self.scene_graph.get_node_by_id(carrier_id)
            if carrier:
                context["carrier_id"] = carrier_id
                context["carrier_node"] = carrier
                context["carrier_label"] = carrier.get("properties", {}).get("label")
                context["carrier_location"] = self._get_node_location(carrier)
                context["carrier_location_label"] = self._get_node_location(carrier, return_label=True)

        # High-priority target
        if params.get("hp_object_id"):
            context["hp_object_id"] = params["hp_object_id"]
            context["hp_target_label"] = params["hp_target_label"]
            context["area_has_hp_target"] = bool(params.get("area_has_hp_target"))
        
        return context
    
    def _get_node_location(self, node: Dict[str, Any], return_label: bool = False) -> Optional[Union[int, str]]:
        """Get node location."""
        location_label = self.scene_graph._get_node_location_label(node)
        if not location_label:
            return None
        if return_label:
            return location_label
        return self.label_to_id_map.get(location_label)
    
    def _get_skill_outcomes(self, skill: str, outcome: str,
                          params: Dict, context: Dict) -> List[Dict]:
        """Get skill outcomes."""
        target_type = None
        if context.get("object_id"):
            node = self.scene_graph.get_node_by_id(context["object_id"])
            target_type = node.get('properties', {}).get('type') if node else None
        
        outcomes = skill_template_manager.get_outcomes(skill, outcome, target_type)
        resolved_outcomes = skill_template_manager.resolve_outcomes(outcomes, params, context)
        return resolved_outcomes
    
    def _start_skill_execution(self, robot_label: str, skill_info: Dict, timestep: int):
        """Record skill execution start."""
        self._current_executions[robot_label] = {
            'robot_label': robot_label,
            'skill_name': skill_info.get('skill'),
            'task_id': skill_info.get('params', {}).get('task_id'),
            'timestep': timestep,
            'start_time': asyncio.get_event_loop().time()
        }
    
    def _end_skill_execution(self, robot_label: str):
        """Record skill execution end."""
        if robot_label in self._current_executions:
            del self._current_executions[robot_label]
    
    def _interrupt_execution(self, event_payload: dict):
        """Interrupt execution with a global stop so all progress waits exit quickly."""
        self._interrupt_payload = event_payload
        self._interrupt_event.set()
    
    def cleanup(self):
        """Clean up resources."""
        if self.visualizer and hasattr(self.visualizer, 'stop_animation'):
            self.visualizer.stop_animation()
        if self.video_writer:
            self.video_writer.close()
        if self.visualizer:
            plt.close(self.visualizer.fig)
