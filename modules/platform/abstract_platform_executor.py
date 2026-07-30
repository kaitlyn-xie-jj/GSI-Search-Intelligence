# -*- coding: utf-8 -*-
"""
Abstract Platform Executor - abstract base class for platform executors

Defines the unified executor interface that all platforms must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable


class AbstractPlatformExecutor(ABC):
    """Abstract base class for platform executors.
    
    This abstract class defines the standard platform executor interface, including:
    - Skill execution interface.
    - Configuration interface.
    - Service lifecycle interface.
    - Context interface.
    - Newcase management interface.
    
    All concrete platform implementations, such as PlatformExecutor and
    UnrealPlatformExecutor, must inherit from this class and implement all abstract methods.
    """
    
    # ==================== Skill Execution Interface ====================
    
    @abstractmethod
    async def execute_plan(
        self, skills_by_timestep: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a skill plan.
        
        Args:
            skills_by_timestep: Skill plan organized by timestep.
                Format: {timestep: {robot_label: {skill: str, params: dict}}}
                
        Returns:
            Execution result dictionary containing:
            - success: Whether execution succeeded.
            - outcomes: Execution result list.
        """
        ...
    
    @abstractmethod
    def set_incremental_outcome_handler(self, handler: Callable) -> None:
        """Set the incremental result handler callback.
        
        Args:
            handler: Async callback used to handle execution results for each timestep.
        """
        ...
    
    @abstractmethod
    def set_knowledge_scope(self, scope: str) -> None:
        """Set the knowledge scope.
        
        Args:
            scope: Knowledge scope, either 'local' or 'global'.
        """
        ...
    
    @abstractmethod
    def set_replay_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        """Set replay metadata.
        
        Args:
            meta: Replay metadata dictionary.
        """
        ...
    
    # ==================== Service Lifecycle Interface ====================
    
    @abstractmethod
    async def start_services(self) -> None:
        """Start background services."""
        ...
    
    @abstractmethod
    async def stop_services(self) -> None:
        """Stop background services."""
        ...
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        ...
    
    @abstractmethod
    def is_services_running(self) -> bool:
        """Check whether services are running.
        
        Returns:
            Whether services are running.
        """
        ...
    
    # ==================== Context Interface ====================
    
    @abstractmethod
    def get_context_hub(self) -> Any:
        """Get the context hub.
        
        Returns:
            Context hub instance.
        """
        ...
    
    @abstractmethod
    async def collect_outcomes(self, drain: bool = True) -> List[Dict[str, Any]]:
        """Collect execution results.
        
        Args:
            drain: Whether to clear stored results.
            
        Returns:
            Execution result list.
        """
        ...
    
    @abstractmethod
    async def await_quiescence(
        self, timeout: float = 1.5, include_state: bool = True
    ) -> bool:
        """Wait for quiescence.
        
        Args:
            timeout: Timeout in seconds.
            include_state: Whether to include state synchronization.
            
        Returns:
            Whether quiescence was reached before timeout.
        """
        ...
    
    # ==================== Newcase Management Interface ====================
    
    @abstractmethod
    def init_newcase_services(
        self,
        n_new_max: int = 5,
        enable_generation: bool = True,
    ) -> None:
        """Initialize newcase services.
        
        Args:
            n_new_max: Maximum number of newcases.
            enable_generation: Whether newcase generation is enabled.
        """
        ...
    
    @abstractmethod
    def get_newcase_controller(self) -> Optional[Any]:
        """Get the newcase controller.
        
        Returns:
            Newcase controller instance, or None if not initialized.
        """
        ...
    
    @abstractmethod
    def get_newcase_injector(self) -> Optional[Any]:
        """Get the newcase injector.
        
        Returns:
            Newcase injector instance, or None if not initialized.
        """
        ...
    
    @abstractmethod
    def get_newcase_generator(self) -> Optional[Any]:
        """Get the newcase generator.
        
        Returns:
            Newcase generator instance, or None if not initialized.
        """
        ...
    
    @abstractmethod
    def get_newcase_summary(self) -> Optional[Dict[str, Any]]:
        """Get the newcase summary.
        
        Returns:
            Newcase summary dictionary, or None if the controller is not initialized.
        """
        ...
    
    @abstractmethod
    async def rollback_last_dynamic_event(self) -> bool:
        """Roll back the last dynamic newcase event.
        
        Returns:
            Whether rollback succeeded.
        """
        ...
