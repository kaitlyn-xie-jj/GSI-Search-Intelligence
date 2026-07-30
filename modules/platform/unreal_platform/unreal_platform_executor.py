# -*- coding: utf-8 -*-
"""
Unreal Platform Executor - Unreal Engine platform executor

Communicates with Unreal Engine over HTTP and implements the AbstractPlatformExecutor interface.
Sends skills to Unreal and handles execution feedback.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING

from modules.platform.abstract_platform_executor import AbstractPlatformExecutor
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.unreal_platform.data_adapters import UnrealDataAdapter
from modules.events.event_bus import publish_event_sync
from modules.config.events import NewCaseEvent

if TYPE_CHECKING:
    from modules.communication.unified_communicator import UnifiedCommunicator


class UnrealPlatformExecutor(AbstractPlatformExecutor):
    """Unreal platform executor implementation.
    
    Communicates with Unreal Engine over HTTP, sends skills, and handles execution feedback.
    
    Attributes:
        _scene_graph: Scene graph instance.
        _communicator: Unified communicator instance.
        _owns_communicator: Whether this instance owns the communicator for cleanup.
        _adapter: Data format adapter.
        _incremental_handler: Incremental result handler callback.
        _knowledge_scope: Knowledge scope.
        _services_running: Service running state.
    """
    
    def __init__(
        self,
        scene_graph: AbstractSceneGraph,
        logger_instance: Optional[logging.Logger] = None,
        communicator: Optional["UnifiedCommunicator"] = None,
        **kwargs
    ):
        """Initialize the Unreal platform executor.
        
        Args:
            scene_graph: Scene graph instance, expected to be UnrealSceneGraph.
            logger_instance: Logger instance.
            communicator: Injected UnifiedCommunicator instance, the recommended approach.
            **kwargs: Additional arguments.
        """
        if scene_graph is None:
            raise RuntimeError("scene_graph cannot be None")
        
        self._scene_graph = scene_graph
        self._logger = logger_instance or logging.getLogger(__name__)
        
        # Use the injected communicator or get it from scene_graph.
        if communicator is not None:
            self._communicator = communicator
            self._owns_communicator = False
        elif hasattr(scene_graph, '_communicator'):
            # Reuse the scene graph communicator.
            self._communicator = scene_graph._communicator
            self._owns_communicator = False
        else:
            # Create a new UnifiedCommunicator instance.
            from modules.communication.unified_communicator import UnifiedCommunicator
            unreal_url = kwargs.get('base_url', 'http://localhost:8080')
            server_port = kwargs.get('server_port', 8081)
            timeout = kwargs.get('timeout', 30.0)
            hitl_enabled = kwargs.get('hitl_enabled', False)
            self._communicator = UnifiedCommunicator(
                unreal_url=unreal_url,
                server_port=server_port,
                timeout=timeout,
                hitl_enabled=hitl_enabled
            )
            self._owns_communicator = True
        
        self._adapter = UnrealDataAdapter()
        
        # Configuration.
        self._incremental_handler: Optional[Callable] = None
        self._knowledge_scope = "local"
        self._replay_meta: Optional[Dict[str, Any]] = None
        
        # Service state.
        self._services_running = False
        
        # The Unreal platform does not need local newcase services.
        # Unreal handles newcase processing.
        self.newcase_controller = None
        self.newcase_injector = None
        self.newcase_generator = None
        
        # Result collection.
        self._collected_outcomes: List[Dict[str, Any]] = []
        self._outcomes_lock = asyncio.Lock()
        
        self._logger.info("UnrealPlatformExecutor initialized")

    # ==================== Skill Execution Interface ====================
    
    async def execute_plan(
        self, skills_by_timestep: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a skill plan.
        
        Sends the skill plan to Unreal Engine for execution and handles feedback.
        The skill list is queued for UE5 polling, then waits for UE5 completion feedback.
        Supports incremental feedback handling: each timestep is handled as soon as feedback arrives.
        
        Args:
            skills_by_timestep: Skill plan organized by timestep.
                Format: {timestep: {robot_label: {skill: str, params: dict}}}
                
        Returns:
            Execution result dictionary containing:
            - success: Whether execution succeeded.
            - outcomes: Execution result list.
        """
        try:
            if not skills_by_timestep:
                self._logger.warning("No skills to execute")
                return {"success": True, "outcomes": []}
            
            self._logger.info(f"Sending {len(skills_by_timestep)} timesteps to Unreal")
            
            # Convert 2D skill parameters to 3D format by adding the z component.
            unreal_skills = self._adapter.skills_to_unreal_format(skills_by_timestep)
            
            # Send to the queue for UE5 polling.
            response = await self._communicator.send_skill_list(unreal_skills)
            execution_id = response.get("execution_id")
            
            if not execution_id:
                self._logger.error("No execution_id returned")
                return {"success": False, "outcomes": [], "message": "No execution_id returned"}
            
            # Collect all outcomes.
            all_outcomes = []

            async def handle_feedback(feedback: Dict[str, Any]):
                """Handle feedback incrementally as each timestep feedback arrives.
                
                Flow:
                1. Identify and publish emergency or info events through the event bus to trigger replanning.
                2. Convert normal feedback to outcomes and pass them to the incremental handler.
                """
                # Process emergency and info events first by publishing them to the event bus.
                await self._process_feedback_events(feedback)

                agent_feedbacks = feedback.get("feedbacks", [])
                step_outcomes = []
                
                for fb in agent_feedbacks:
                    # Emergency feedback is handled through the event bus, so do not emit a duplicate outcome.
                    if self._adapter.is_emergency_feedback(fb):
                        continue

                    outcome = self._adapter.single_feedback_to_outcome(fb)
                    if outcome:
                        all_outcomes.append(outcome)
                        step_outcomes.append(outcome)
                
                # Sync world state after each timestep.
                if step_outcomes:
                    try:
                        await self._scene_graph.sync_from_unreal()
                    except Exception as e:
                        self._logger.warning(f"Failed to sync world state after timestep: {e}")
                    
                    # Call the incremental handler.
                    if self._incremental_handler:
                        try:
                            await self._incremental_handler(step_outcomes)
                        except Exception as e:
                            self._logger.error(f"Error in incremental handler: {e}")
            
            # Wait for execution completion while handling feedback incrementally.
            final_result = await self._communicator.wait_for_completion(
                execution_id,
                on_feedback=handle_feedback
            )
            
            # Sync the latest world state.
            try:
                await self._scene_graph.sync_from_unreal()
            except Exception as e:
                self._logger.warning(f"Failed to sync world state after execution: {e}")
            
            # Store results.
            async with self._outcomes_lock:
                self._collected_outcomes.extend(all_outcomes)
            
            success = final_result.get("success", False)
            self._logger.info(f"Execution {execution_id} completed: success={success}, outcomes={len(all_outcomes)}")
            
            return {
                "success": success,
                "outcomes": all_outcomes,
                "execution_id": execution_id
            }
            
        except TimeoutError as e:
            self._logger.error(f"Execution timed out: {e}")
            return {"success": False, "outcomes": [], "message": str(e)}
        except Exception as e:
            self._logger.error(f"Error executing plan: {e}")
            return {"success": False, "outcomes": [], "message": str(e)}
    
    def set_incremental_outcome_handler(self, handler: Callable) -> None:
        """Set the incremental result handler callback."""
        self._incremental_handler = handler
    
    def set_knowledge_scope(self, scope: str) -> None:
        """Set the knowledge scope."""
        self._knowledge_scope = scope
    
    def set_replay_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        """Set replay metadata."""
        self._replay_meta = meta

    # ==================== Emergency Event Publishing ====================

    async def _publish_new_case_event(
        self, case_id: str, description: Any, entity_id: str,
        entity_type: str, priority: int
    ) -> None:
        """Publish a NewCaseEvent to the global event bus.
        
        Args:
            case_id: Unique event ID.
            description: Event description dict with phase, skill, reason, details, robot, etc.
            entity_id: Associated entity ID, usually robot_id.
            entity_type: Entity type, usually "robot".
            priority: Event priority.
        """
        new_case_event = NewCaseEvent(
            case_id=case_id,
            description=description,
            entity_id=str(entity_id or ""),
            entity_type=str(entity_type or ""),
            timestamp=datetime.now(),
        )
        await publish_event_sync(new_case_event)
        self._logger.warning(f"[UnrealPlatform] NewCaseEvent published: {description.get('reason', '')}")

    async def _process_feedback_events(self, feedback: Dict[str, Any]) -> None:
        """Identify and publish emergency and info events from UE5 feedback.
        
        Logic:
        1. If feedback contains is_emergency_event=true, publish it as a NewCaseEvent to trigger replanning.
        2. If normal feedback contains info_events, publish them as info-level NewCaseEvents.
        
        Args:
            feedback: UE5 timestep feedback containing the feedbacks array.
        """
        agent_feedbacks = feedback.get("feedbacks", [])

        for fb in agent_feedbacks:
            # 1) Emergency event from a precheck or runtime check failure.
            if self._adapter.is_emergency_feedback(fb):
                description = self._adapter.extract_emergency_description(fb)
                robot_id = description.get("robot", {}).get("id", "")
                reason = description.get("reason", "unknown")
                case_id = f"unreal-{reason}-{robot_id}"
                await self._publish_new_case_event(
                    case_id=case_id,
                    description=description,
                    entity_id=str(robot_id),
                    entity_type="robot",
                    priority=1,
                )
                continue  # Do not process emergency feedback as a normal outcome.

            # 2) Info events in normal feedback.
            info_descriptions = self._adapter.extract_info_events(fb)
            for idx, desc in enumerate(info_descriptions):
                robot_id = desc.get("robot", {}).get("id", "")
                reason = desc.get("reason", "info")
                case_id = f"unreal-info-{reason}-{robot_id}-{idx}"
                await self._publish_new_case_event(
                    case_id=case_id,
                    description=desc,
                    entity_id=str(robot_id),
                    entity_type="robot",
                    priority=3,
                )

    # ==================== Service Lifecycle Interface ====================
    
    async def start_services(self) -> None:
        """Start background services.
        
        Starts the HTTP server for UE5 polling and checks whether the UE5 API is available.
        The server runs in a separate thread and does not block the main asyncio event loop.
        
        Note: the HTTP server may already be started in initialize_platform().
        start_server() is idempotent and skips an already running server.
        """
        if self._services_running:
            self._logger.warning("Services already running")
            return
        
        try:
            # Start the HTTP server in a separate thread for UE5 skill-list polling.
            # This is idempotent and returns directly if the server is already running.
            await self._communicator.start_server()
            self._logger.info("HTTP server ready for UE5 polling")
            
            # Check whether the UE5 API is available.
            is_healthy = await self._communicator.health_check()
            if not is_healthy:
                self._logger.warning("Unreal API health check failed, but continuing...")
            else:
                self._logger.info("Unreal API health check passed")
            
            self._services_running = True
            self._logger.info("Unreal platform services started successfully")
        except Exception as e:
            self._logger.error(f"Failed to start services: {e}")
            # Mark as running even on failure to avoid repeated attempts.
            self._services_running = True
    
    async def stop_services(self) -> None:
        """Stop background services.
        
        Gracefully shuts down the HTTP server and releases resources.
        Closes the communicator only when this instance owns it.
        """
        if not self._services_running:
            return
        
        try:
            # Stop the server only when this instance owns the communicator.
            if self._owns_communicator:
                # Stop the HTTP server used for UE5 polling.
                await self._communicator.stop_server()
                
                # Close the HTTP client session used to access UE5.
                await self._communicator.close()
            
            self._services_running = False
            self._logger.info("Unreal platform services stopped")
        except Exception as e:
            self._logger.error(f"Error stopping services: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear collected results.
            self._collected_outcomes.clear()
            self._logger.info("Unreal platform resources cleaned up")
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")
    
    def is_services_running(self) -> bool:
        """Check whether services are running."""
        return self._services_running

    # ==================== Context Interface ====================
    
    def get_context_hub(self) -> Any:
        """Get the context hub.
        
        The Unreal platform has no local context hub, so this returns None.
        """
        return None
    
    async def collect_outcomes(self, drain: bool = True) -> List[Dict[str, Any]]:
        """Collect execution results."""
        async with self._outcomes_lock:
            outcomes = self._collected_outcomes.copy()
            if drain:
                self._collected_outcomes.clear()
            return outcomes
    
    async def await_quiescence(
        self, timeout: float = 1.5, include_state: bool = True
    ) -> bool:
        """Wait for quiescence.
        
        For the Unreal platform, sync the latest state and then return.
        """
        try:
            if include_state:
                await self._scene_graph.sync_from_unreal()
            return True
        except Exception as e:
            self._logger.warning(f"Error during quiescence: {e}")
            return False

    # ==================== Newcase Management Interface ====================
    # UE handles newcase processing; local methods return None or no-op.
    
    def init_newcase_services(
        self,
        n_new_max: int = 5,
        enable_generation: bool = True,
    ) -> None:
        """Initialize newcase services.
        
        Newcase for the Unreal platform is handled by Unreal, so no local initialization is needed.
        """
        self._logger.info("Newcase services are handled by Unreal, skipping local init")
    
    def get_newcase_controller(self) -> Optional[Any]:
        """Get the newcase controller.
        
        The Unreal platform returns None because Unreal handles newcase processing.
        """
        return None
    
    def get_newcase_injector(self) -> Optional[Any]:
        """Get the newcase injector.
        
        The Unreal platform returns None because Unreal handles newcase processing.
        """
        return None
    
    def get_newcase_generator(self) -> Optional[Any]:
        """Get the newcase generator.
        
        The Unreal platform returns None because Unreal handles newcase processing.
        """
        return None
    
    def get_newcase_summary(self) -> Optional[Dict[str, Any]]:
        """Get the newcase summary.
        
        The Unreal platform returns None because Unreal handles newcase processing.
        """
        return None
    
    async def rollback_last_dynamic_event(self) -> bool:
        """Roll back the last dynamic newcase event.
        
        The Unreal platform does not support local rollback, so this returns False.
        """
        self._logger.warning("Rollback not supported for Unreal platform")
        return False

    # ==================== Context Manager Support ====================
    
    async def __aenter__(self) -> 'UnrealPlatformExecutor':
        """Enter the async context manager."""
        await self.start_services()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        await self.stop_services()
        self.cleanup()
    
    def __repr__(self) -> str:
        return (
            f"UnrealPlatformExecutor("
            f"services_running={self._services_running}, "
            f"knowledge_scope={self._knowledge_scope})"
        )
