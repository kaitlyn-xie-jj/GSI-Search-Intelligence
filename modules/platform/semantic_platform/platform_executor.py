# -*- coding: utf-8 -*-
"""
Platform Executor - unified entry point for the platform execution layer.

Wraps the scene graph manager, skill executor, and related services to provide a unified platform-layer API.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable

from modules.platform.abstract_platform_executor import AbstractPlatformExecutor
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.semantic_platform.scene_graph_manager import SemanticSceneGraph
from modules.platform.semantic_platform.skill_executor import SkillExecutor
from modules.platform.semantic_platform.context_hub import ContextHub
from modules.platform.semantic_platform.new_case_controller import NewCaseController
from modules.platform.semantic_platform.new_case_injector import NewCaseInjector
from modules.platform.semantic_platform.new_case_generator import NewCaseGenerator
from modules.config.entities.new_case_config import NEW_CASE_OP_TEMPLATES
from modules.config.system_config import config

logger = logging.getLogger(__name__)


class PlatformExecutor(AbstractPlatformExecutor):
    """
    Unified entry point for the platform execution layer.
    Wraps the scene graph, skill executor, and related services.
    
    This class provides the unified platform-layer API, including:
    - Scene graph access API
    - Skill execution API
    - New-case management API
    - Context hub API
    - Service lifecycle management
    
    Attributes:
        scene_graph: Scene graph manager instance.
        skill_executor: Skill executor instance.
        context_hub: Context hub instance.
        newcase_controller: New-case controller instance (optional).
        newcase_injector: New-case injector instance (optional).
        newcase_generator: New-case generator instance (optional).
    """
    
    def __init__(
        self,
        scene_graph: AbstractSceneGraph,
        enable_visualization: bool = False,
        enable_video_recording: bool = False,
        video_output_path: Optional[str] = None,
        logger_instance: Optional[logging.Logger] = None,
        is_replay: bool = False,
    ):
        """
        Initialize the platform executor.
        
        Args:
            scene_graph: Scene graph manager instance.
            enable_visualization: Whether to enable visualization.
            enable_video_recording: Whether to enable video recording.
            video_output_path: Video output path.
            logger_instance: Logger instance.
            is_replay: Whether replay mode is enabled.
        """
        if scene_graph is None:
            raise RuntimeError("scene_graph cannot be None")
        
        self._logger = logger_instance or logger
        self._is_replay = is_replay
        
        # Core component: scene graph manager
        self.scene_graph = scene_graph
        
        # Core component: skill executor
        self.skill_executor = SkillExecutor(
            scene_graph=scene_graph,
            enable_visualization=enable_visualization,
            enable_video_recording=enable_video_recording,
            video_output_path=video_output_path,
            logger=self._logger,
            is_replay=is_replay,
        )
        
        # Core component: context hub
        self.context_hub = ContextHub(
            scene_graph=scene_graph,
        )
        
        # Set the context hub on the skill executor
        self.skill_executor.set_context_hub(self.context_hub)
        
        # New-case handling components
        self.newcase_controller: Optional[NewCaseController] = None
        self.newcase_injector: Optional[NewCaseInjector] = None
        self.newcase_generator: Optional[NewCaseGenerator] = None

        n_new = config.max_newcases_per_run or 5
        self.init_newcase_services(
            n_new_max=n_new,
            enable_generation=config.enable_new_case_generation,
        )
        
        # Service runtime state
        self._services_running = False
        
        self._logger.info("PlatformExecutor initialized")


    # ==================== Skill Execution API ====================
    
    async def execute_plan(self, skills_by_timestep: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a complete skill plan.
        
        Args:
            skills_by_timestep: Skill plan organized by timestep.
                Format: {timestep: {robot_label: {skill: str, params: dict}}}
                
        Returns:
            Execution result dictionary, including:
            - success: Whether execution succeeded.
            - outcomes: Execution result list.
        """
        return await self.skill_executor.execute_plan(skills_by_timestep)
    
    def set_incremental_outcome_handler(self, handler: Callable) -> None:
        """
        Set the incremental outcome handler.
        
        Args:
            handler: Async callback for handling execution results at each timestep.
        """
        self.skill_executor.set_incremental_outcome_handler(handler)
    
    def set_knowledge_scope(self, scope: str) -> None:
        """
        Set the knowledge scope.
        
        Args:
            scope: Knowledge scope ('local' or 'global').
        """
        self.skill_executor.set_knowledge_scope(scope)
    
    def set_replay_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        """
        Set replay metadata.
        
        Args:
            meta: Replay metadata dictionary.
        """
        self.skill_executor.set_replay_meta(meta)
        if self.newcase_controller:
            self.newcase_controller.set_replay_meta(meta)
        if self.newcase_generator:
            self.newcase_generator.set_replay_meta(meta)

    # ==================== New-Case Management API ====================
    
    def init_newcase_services(
        self,
        n_new_max: int = 5,
        enable_generation: bool = True,
    ) -> None:
        """
        Initialize new-case services.
        
        Args:
            n_new_max: Maximum number of new cases.
            enable_generation: Whether to enable new-case generation.
        """
        # Initialize only when new-case generation is enabled
        if not config.enable_new_case_generation:
            self._logger.info("New case generation is disabled by config")
            return
        
        # Initialize new-case controller
        self.newcase_controller = NewCaseController(
            n_new_max=n_new_max,
            enable_generation=enable_generation,
            is_replay=self._is_replay,
            spacing_factor=config.get_config('newcase_spacing_factor', 2.0),
            cooldown_rounds=config.get_config('newcase_cooldown_rounds', 2),
            similarity_threshold=config.get_config('newcase_similarity_threshold', 0.8),
            similarity_damping=config.get_config('newcase_similarity_damping', 0.3),
        )
        
        # Initialize new-case injector
        self.newcase_injector = NewCaseInjector(
            scene_graph=self.scene_graph,
            is_replay=self._is_replay,
        )
        
        # Initialize new-case generator
        self.newcase_generator = NewCaseGenerator(
            templates=NEW_CASE_OP_TEMPLATES,
            scene_graph=self.scene_graph,
            max_events=n_new_max,
            is_replay=self._is_replay,
        )
        
        # Set new-case services on the skill executor
        self.skill_executor.set_new_case_services(
            generator=self.newcase_generator,
            injector=self.newcase_injector,
            ctrl=self.newcase_controller,
        )
        
        self._logger.info(
            f"New case services initialized: n_new_max={n_new_max}, "
            f"enable_generation={enable_generation}"
        )
    
    def get_newcase_controller(self) -> Optional[NewCaseController]:
        """
        Get the new-case controller.
        
        Returns:
            New-case controller instance, or None if not initialized.
        """
        return self.newcase_controller
    
    def get_newcase_injector(self) -> Optional[NewCaseInjector]:
        """
        Get the new-case injector.
        
        Returns:
            New-case injector instance, or None if not initialized.
        """
        return self.newcase_injector
    
    def get_newcase_generator(self) -> Optional[NewCaseGenerator]:
        """
        Get the new-case generator.
        
        Returns:
            New-case generator instance, or None if not initialized.
        """
        return self.newcase_generator
    
    def get_newcase_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get the new-case stats summary.
        
        Returns:
            New-case stats summary dictionary, or None if the controller is not initialized.
        """
        if self.newcase_controller:
            return self.newcase_controller.summary()
        return None
    
    async def rollback_last_dynamic_event(self) -> bool:
        """
        Roll back the last dynamic new-case event.
        
        Returns:
            Whether rollback succeeded.
        """
        if self.newcase_injector:
            return await self.newcase_injector.rollback_last_dynamic_event()
        return False

    # ==================== Context Hub API ====================
    
    def get_context_hub(self) -> ContextHub:
        """
        Get the context hub.
        
        Returns:
            Context hub instance.
        """
        return self.context_hub
    
    async def collect_outcomes(self, drain: bool = True) -> List[Dict[str, Any]]:
        """
        Collect accumulated execution outcomes.
        
        Args:
            drain: Whether to clear stored results.
            
        Returns:
            Execution result list.
        """
        return await self.context_hub.collect_outcomes(drain=drain)
    
    async def await_quiescence(self, timeout: float = 1.5, include_state: bool = True) -> bool:
        """
        Wait for the context hub to reach quiescence.
        
        Args:
            timeout: Timeout in seconds.
            include_state: Whether to include state sync.
            
        Returns:
            Whether quiescence was reached before timeout.
        """
        return await self.context_hub.await_quiescence(timeout=timeout, include_state=include_state)
    
    # ==================== Service Lifecycle Management ====================
    
    async def start_services(self) -> None:
        """
        Start background services.
        
        Starts context hub background tasks, including the state receiver,
        operation processor, and result aggregator.
        """
        if self._services_running:
            self._logger.warning("Services already running")
            return
        
        try:
            # Start context hub services
            await self.context_hub.start()
            self._services_running = True
            self._logger.info("Platform services started")
        except Exception as e:
            self._logger.error(f"Failed to start services: {e}", exc_info=True)
            raise
    
    async def stop_services(self) -> None:
        """
        Stop background services.
        
        Stops context hub background tasks and waits for all tasks to complete.
        """
        if not self._services_running:
            return
        
        try:
            # Stop context hub services
            await self.context_hub.stop()
            
            # Stop new-case injector
            if self.newcase_injector:
                self.newcase_injector.stop()
            
            self._services_running = False
            self._logger.info("Platform services stopped")
        except Exception as e:
            self._logger.error(f"Error stopping services: {e}", exc_info=True)
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        Cleans up skill executor visualization resources and the video recorder.
        """
        try:
            # Clean up skill executor resources
            self.skill_executor.cleanup()
            self._logger.info("Platform resources cleaned up")
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def is_services_running(self) -> bool:
        """
        Check whether services are running.
        
        Returns:
            Whether services are running.
        """
        return self._services_running
    
    # ==================== Context Manager Support ====================
    
    async def __aenter__(self) -> 'PlatformExecutor':
        """
        Async context manager entry.
        
        Returns:
            PlatformExecutor instance.
        """
        await self.start_services()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit.
        """
        await self.stop_services()
        self.cleanup()
