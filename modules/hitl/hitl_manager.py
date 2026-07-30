"""
HITL Manager - Global Singleton for Human-in-the-Loop Interactions.

This module provides the HITLManager class, a thread-safe global singleton
that manages all human-in-the-loop interactions in the system.
It is typically imported via the package's __init__.py.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.hitl.config import HITLConfig, InteractionState
from modules.communication.enums import (
    MessageCategory,
    MessageDirection,
    ReviewType,
    DecisionType,
)
from modules.communication.message import HITLMessage


logger = logging.getLogger(__name__)


@dataclass
class HITLStatistics:
    """
    Statistics for HITL interactions.
    
    Tracks counts, success rates, and response times for all interaction types.
    """
    # Instruction statistics
    instruction_count: int = 0
    instruction_success_count: int = 0
    instruction_timeout_count: int = 0
    instruction_response_times: List[float] = field(default_factory=list)
    
    # Review statistics
    review_count: int = 0
    review_success_count: int = 0
    review_timeout_count: int = 0
    review_modified_count: int = 0
    review_response_times: List[float] = field(default_factory=list)
    
    # Decision statistics
    decision_count: int = 0
    decision_success_count: int = 0
    decision_timeout_count: int = 0
    decision_response_times: List[float] = field(default_factory=list)
    decision_by_option: Dict[str, int] = field(default_factory=dict)
    
    def get_instruction_success_rate(self) -> float:
        """Get instruction success rate."""
        if self.instruction_count == 0:
            return 0.0
        return self.instruction_success_count / self.instruction_count
    
    def get_review_success_rate(self) -> float:
        """Get review success rate."""
        if self.review_count == 0:
            return 0.0
        return self.review_success_count / self.review_count
    
    def get_decision_success_rate(self) -> float:
        """Get decision success rate."""
        if self.decision_count == 0:
            return 0.0
        return self.decision_success_count / self.decision_count
    
    def get_avg_instruction_response_time(self) -> float:
        """Get average instruction response time."""
        if not self.instruction_response_times:
            return 0.0
        return sum(self.instruction_response_times) / len(self.instruction_response_times)
    
    def get_avg_review_response_time(self) -> float:
        """Get average review response time."""
        if not self.review_response_times:
            return 0.0
        return sum(self.review_response_times) / len(self.review_response_times)
    
    def get_avg_decision_response_time(self) -> float:
        """Get average decision response time."""
        if not self.decision_response_times:
            return 0.0
        return sum(self.decision_response_times) / len(self.decision_response_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "instruction": {
                "count": self.instruction_count,
                "success_count": self.instruction_success_count,
                "timeout_count": self.instruction_timeout_count,
                "success_rate": self.get_instruction_success_rate(),
                "avg_response_time": self.get_avg_instruction_response_time(),
            },
            "review": {
                "count": self.review_count,
                "success_count": self.review_success_count,
                "timeout_count": self.review_timeout_count,
                "modified_count": self.review_modified_count,
                "success_rate": self.get_review_success_rate(),
                "avg_response_time": self.get_avg_review_response_time(),
            },
            "decision": {
                "count": self.decision_count,
                "success_count": self.decision_success_count,
                "timeout_count": self.decision_timeout_count,
                "success_rate": self.get_decision_success_rate(),
                "avg_response_time": self.get_avg_decision_response_time(),
                "by_option": dict(self.decision_by_option),
            },
        }
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.instruction_count = 0
        self.instruction_success_count = 0
        self.instruction_timeout_count = 0
        self.instruction_response_times = []
        self.review_count = 0
        self.review_success_count = 0
        self.review_timeout_count = 0
        self.review_modified_count = 0
        self.review_response_times = []
        self.decision_count = 0
        self.decision_success_count = 0
        self.decision_timeout_count = 0
        self.decision_response_times = []
        self.decision_by_option = {}


class HITLManager:
    """
    Human-in-the-Loop Interaction Manager - Global Singleton.
    
    This class manages all HITL interactions including:
    - Instruction input from operators
    - Plan review and modification
    - Decision requests for ambiguous situations
    
    The manager is implemented as a thread-safe singleton to ensure
    consistent state across all components of the system.
    
    Usage:
        manager = get_hitl_manager()
        manager.initialize(config, communicator)
        
        if manager.is_instruction_enabled:
            instruction = await manager.wait_for_instruction()
    """
    
    _instance: Optional['HITLManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'HITLManager':
        """
        Create or return the singleton instance.
        
        Thread-safe implementation using double-checked locking.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self) -> None:
        """
        Initialize the manager instance.
        
        This is called every time HITLManager() is invoked, but
        actual initialization only happens once due to _initialized flag.
        """
        if getattr(self, '_initialized', False):
            return
        
        self._config = HITLConfig()
        self._state = InteractionState()
        self._communicator: Optional[Any] = None  # UnifiedCommunicator
        self._state_lock = threading.Lock()
        self._statistics = HITLStatistics()
        self._initialized = True
        
        logger.debug("HITLManager instance created")
    
    def initialize(
        self,
        config: Dict[str, Any],
        communicator: Optional[Any] = None
    ) -> None:
        """
        Initialize the manager with configuration and communicator.
        
        Args:
            config: Full configuration dictionary containing 'human_in_loop' section
            communicator: UnifiedCommunicator instance for message handling
        """
        hitl_config = config.get("human_in_loop", {})
        self._config = HITLConfig.from_dict(hitl_config)
        self._communicator = communicator
        
        # Reset state and statistics on re-initialization
        with self._state_lock:
            self._state = InteractionState()
            self._statistics = HITLStatistics()
        
        # Register message handlers if communicator is provided
        if self._communicator and self._config.enabled:
            self._register_handlers()
        
        logger.info(
            f"HITLManager initialized: enabled={self._config.enabled}, "
            f"instruction={self._config.instruction_enabled}, "
            f"review={self._config.review_enabled}, "
            f"decision={self._config.decision_enabled}"
        )
    
    def _register_handlers(self) -> None:
        """Register message handlers with the communicator."""
        if not self._communicator:
            return
        
        self._communicator.register_hitl_handler(
            MessageCategory.INSTRUCTION,
            self._handle_instruction_message
        )
        self._communicator.register_hitl_handler(
            MessageCategory.REVIEW,
            self._handle_review_message
        )
        self._communicator.register_hitl_handler(
            MessageCategory.DECISION,
            self._handle_decision_message
        )
        
        logger.debug("HITL message handlers registered")
    
    # ==================== Properties ====================
    
    @property
    def is_enabled(self) -> bool:
        """Check if HITL mode is enabled."""
        return self._config.enabled
    
    @property
    def is_instruction_enabled(self) -> bool:
        """
        Check if instruction input is enabled.
        
        Returns False if master enabled flag is False.
        """
        return self._config.enabled and self._config.instruction_enabled
    
    @property
    def is_review_enabled(self) -> bool:
        """
        Check if plan review is enabled.
        
        Returns False if master enabled flag is False.
        """
        return self._config.enabled and self._config.review_enabled
    
    @property
    def is_decision_enabled(self) -> bool:
        """
        Check if decision requests are enabled.
        
        Returns False if master enabled flag is False.
        """
        return self._config.enabled and self._config.decision_enabled
    
    @property
    def config(self) -> HITLConfig:
        """Get the current configuration."""
        return self._config
    
    @property
    def state(self) -> InteractionState:
        """Get the current interaction state."""
        return self._state
    
    @property
    def communicator(self) -> Optional[Any]:
        """Get the communicator instance."""
        return self._communicator
    
    @property
    def statistics(self) -> HITLStatistics:
        """Get the interaction statistics."""
        return self._statistics
    
    # ==================== Message Handlers ====================
    
    def _handle_instruction_message(self, message: HITLMessage) -> None:
        """
        Handle incoming instruction messages.
        
        Note: With UnifiedCommunicator, messages are automatically placed
        in the inbound queue by the HTTP handler.
        """
        logger.debug(f"Instruction message received: {message.message_id}")
    
    def _handle_review_message(self, message: HITLMessage) -> None:
        """
        Handle incoming review response messages.
        
        Note: With UnifiedCommunicator, messages are automatically placed
        in the inbound queue by the HTTP handler.
        """
        logger.debug(f"Review response received: {message.message_id}")
    
    def _handle_decision_message(self, message: HITLMessage) -> None:
        """
        Handle incoming decision response messages.
        
        Note: With UnifiedCommunicator, messages are automatically placed
        in the inbound queue by the HTTP handler.
        """
        logger.debug(f"Decision response received: {message.message_id}")

    # ==================== Instruction Interaction ====================
    
    async def request_instruction(self) -> bool:
        """
        Send instruction request to UE5.
        
        This method sends a user_instruction message to UE5,
        prompting the user to input an instruction.
        
        Returns:
            True if request was sent successfully, False otherwise.
        """
        if not self.is_instruction_enabled:
            logger.debug("Instruction input disabled, not sending request")
            return False
        
        if not self._communicator:
            logger.warning("No communicator available for instruction request")
            return False
        
        try:
            # Build and send instruction request message
            message = HITLMessage.create(
                category=MessageCategory.INSTRUCTION,
                msg_type="user_instruction",
                direction=MessageDirection.PYTHON_TO_UE5,
                payload={
                    "request_type": "instruction",
                    "prompt": "Please enter your instruction"
                }
            )
            self._communicator.send_hitl_message(message)
            
            logger.info(
                f"[HITL] Instruction request sent: message_id={message.message_id}, "
                f"direction={message.direction.value}, timestamp={message.timestamp}"
            )
            return True
            
        except Exception as e:
            logger.error(f"[HITL] Error sending instruction request: {e}")
            return False
    
    async def wait_for_instruction(self) -> Optional[Dict[str, Any]]:
        """
        Request and wait for user instruction input from UE5.
        
        This method first sends a user_instruction message to UE5
        to prompt the user for input, then blocks until an instruction
        message is received or timeout occurs. If instruction is disabled,
        returns None immediately without blocking.
        
        Returns:
            Instruction payload dictionary if received, None if disabled
            or timeout occurs.
        """
        if not self.is_instruction_enabled:
            logger.debug("Instruction input disabled, returning None immediately")
            return None
        
        if not self._communicator:
            logger.warning("No communicator available for instruction input")
            return None
        
        with self._state_lock:
            self._state.instruction_pending = True
            self._statistics.instruction_count += 1
        
        start_time = time.time()
        
        try:
            # First, send instruction request to UE5
            request_sent = await self.request_instruction()
            if not request_sent:
                logger.warning("[HITL] Failed to send instruction request")
            
            logger.info(
                f"[HITL] Waiting for instruction input... "
                f"(timeout={self._config.instruction_timeout}s)"
            )
            response = await self._communicator.wait_for_hitl_response(
                MessageCategory.INSTRUCTION,
                timeout=self._config.instruction_timeout
            )
            
            response_time = time.time() - start_time
            
            if response:
                with self._state_lock:
                    self._state.last_instruction = response.payload
                    self._statistics.instruction_success_count += 1
                    self._statistics.instruction_response_times.append(response_time)
                
                logger.info(
                    f"[HITL] Instruction received: message_id={response.message_id}, "
                    f"direction={response.direction.value}, "
                    f"timestamp={response.timestamp}, "
                    f"response_time={response_time:.2f}s"
                )
                return response.payload
            
            with self._state_lock:
                self._statistics.instruction_timeout_count += 1
            
            logger.warning(
                f"[HITL] Instruction timeout after {self._config.instruction_timeout}s, "
                f"falling back to pre-loaded instruction"
            )
            return None
            
        except Exception as e:
            logger.error(f"[HITL] Error waiting for instruction: {e}")
            return None
            
        finally:
            with self._state_lock:
                self._state.instruction_pending = False
    
    # ==================== Review Interaction ====================
    
    async def request_review(
        self,
        review_type: ReviewType,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Request plan review from the operator.
        
        Sends the data to UE5 for review and waits for the operator's
        response. If review is disabled, returns the original data
        immediately without blocking.
        
        Args:
            review_type: Type of review (TASK_GRAPH or SKILL_LIST)
            data: The data to be reviewed
            
        Returns:
            Modified data if operator made changes, original data otherwise.
        """
        
        if not self.is_review_enabled:
            logger.debug("Review disabled, returning original data")
            return data
        
        if not self._communicator:
            logger.warning("No communicator available for review")
            return data
        
        with self._state_lock:
            self._state.review_pending = True
            self._statistics.review_count += 1
        
        start_time = time.time()
        
        try:
            # Build and send review request message
            message = HITLMessage.create(
                category=MessageCategory.REVIEW,
                msg_type=review_type.value,
                direction=MessageDirection.PYTHON_TO_UE5,
                payload={
                    "review_type": review_type.value,
                    "data": data
                }
            )
            self._communicator.send_hitl_message(message)
            
            logger.info(
                f"[HITL] Review request sent: message_id={message.message_id}, "
                f"type={review_type.value}, direction={message.direction.value}, "
                f"timestamp={message.timestamp}"
            )
            
            response = await self._communicator.wait_for_hitl_response(
                MessageCategory.REVIEW,
                timeout=self._config.review_timeout
            )
            
            response_time = time.time() - start_time
            
            if response:
                with self._state_lock:
                    self._state.last_review_result = response.payload
                    self._statistics.review_success_count += 1
                    self._statistics.review_response_times.append(response_time)
                    self._statistics.review_modified_count += 1

                    modified_data = response.payload.get("data", data)
                    logger.info(
                        f"[HITL] Review completed: "
                        f"message_id={response.message_id}, "
                        f"response_time={response_time:.2f}s"
                    )
                    return modified_data
            
            with self._state_lock:
                self._statistics.review_timeout_count += 1
            
            logger.warning(
                f"[HITL] Review timeout after {self._config.review_timeout}s, "
                f"proceeding with original data"
            )
            return data
            
        except Exception as e:
            logger.error(f"[HITL] Error during review: {e}")
            import traceback
            traceback.print_exc()
            return data
            
        finally:
            with self._state_lock:
                self._state.review_pending = False
    
    # ==================== Decision Interaction ====================
    
    async def request_decision(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Request a decision from the operator.
        
        Sends the decision context and description to UE5 and
        waits for the operator's selection. If decision is disabled,
        returns the default option immediately.
        
        Args:
            decision_type: Type of decision (SEARCH_NOT_FOUND, etc.)
            context: Context information for the decision
            description: Human-readable description text for the decision
            
        Returns:
            Dictionary with decision and optional user_feedback.
        """
        default_response = {"decision": "end_task"}
        
        if not self.is_decision_enabled:
            logger.debug("Decision disabled, returning default option")
            return default_response
        
        if not self._communicator:
            logger.warning("No communicator available for decision")
            return default_response
        
        with self._state_lock:
            self._state.decision_pending = True
            self._statistics.decision_count += 1
        
        start_time = time.time()
        
        try:
            # Build and send decision request message
            message = HITLMessage.create(
                category=MessageCategory.DECISION,
                msg_type=decision_type.value,
                direction=MessageDirection.PYTHON_TO_UE5,
                payload={
                    "decision_type": decision_type.value,
                    "description": description,
                    "context": context,
                }
            )
            self._communicator.send_hitl_message(message)
            
            logger.info(
                f"[HITL] Decision request sent: message_id={message.message_id}, "
                f"type={decision_type.value}, direction={message.direction.value}, "
                f"timestamp={message.timestamp}, "
                f"context={context}"
            )
            
            # Wait for decision response
            response = await self._communicator.wait_for_hitl_response(
                MessageCategory.DECISION,
                timeout=self._config.decision_timeout
            )
            
            response_time = time.time() - start_time
            
            if response:
                payload = dict(response.payload)

                with self._state_lock:
                    self._state.last_decision = payload
                    self._statistics.decision_success_count += 1
                    self._statistics.decision_response_times.append(response_time)
                
                selected = payload.get("decision", "end_task")
                
                # Track decision by option
                with self._state_lock:
                    self._statistics.decision_by_option[selected] = self._statistics.decision_by_option.get(selected, 0) + 1
                
                logger.info(
                    f"[HITL] Decision received: message_id={response.message_id}, "
                    f"decision={selected}, "
                    f"context={context}, "
                    f"response_time={response_time:.2f}s"
                )
                return payload
            
            with self._state_lock:
                self._statistics.decision_timeout_count += 1
            
            logger.warning(
                f"[HITL] Decision timeout after {self._config.decision_timeout}s, "
                f"defaulting to end_task"
            )
            return default_response
            
        except Exception as e:
            logger.error(f"[HITL] Error during decision: {e}")
            return default_response
            
        finally:
            with self._state_lock:
                self._state.decision_pending = False
    
    # ==================== Error Handling and Recovery ====================
    
    def clear_pending_states(self) -> None:
        """
        Clear all pending interaction states.
        
        This method should be called during error recovery to reset
        the interaction state machine and allow new interactions.
        """
        with self._state_lock:
            self._state.clear_pending_states()
        logger.info("All pending interaction states cleared")
    
    async def _retry_with_fallback(
        self,
        operation: str,
        coro,
        fallback_value: Any
    ) -> Any:
        """
        Execute an operation with retry logic and fallback.
        
        Args:
            operation: Name of the operation for logging
            coro: Coroutine to execute
            fallback_value: Value to return if all retries fail
            
        Returns:
            Result of the operation or fallback value.
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self._config.retry_count):
            try:
                return await coro
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{operation} failed (attempt {attempt + 1}/"
                    f"{self._config.retry_count}): {e}"
                )
                
                if attempt < self._config.retry_count - 1:
                    import asyncio
                    await asyncio.sleep(self._config.retry_delay)
        
        logger.error(
            f"{operation} failed after {self._config.retry_count} attempts, "
            f"falling back to autonomous mode. Last error: {last_error}"
        )
        
        # Clear pending states on failure
        self.clear_pending_states()
        
        return fallback_value
    
    def reset(self) -> None:
        """
        Reset the manager to initial state.
        
        This clears all state, configuration, and statistics, useful for testing
        or when reinitializing the system.
        """
        with self._state_lock:
            self._config = HITLConfig()
            self._state = InteractionState()
            self._statistics = HITLStatistics()
            self._communicator = None
        logger.info("HITLManager reset to initial state")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get the current interaction statistics as a dictionary.
        
        Returns:
            Dictionary containing all interaction statistics.
        """
        with self._state_lock:
            return self._statistics.to_dict()
    
    def reset_statistics(self) -> None:
        """Reset all interaction statistics."""
        with self._state_lock:
            self._statistics.reset()
        logger.info("HITL statistics reset")
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.
        
        This is primarily useful for testing to ensure a fresh instance.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._initialized = False
                cls._instance = None
        logger.debug("HITLManager singleton instance reset")


def get_hitl_manager() -> HITLManager:
    """
    Get the global HITLManager instance.
    
    This is the recommended way to access the HITLManager singleton.
    
    Returns:
        The global HITLManager instance.
    """
    return HITLManager()
