# -*- coding: utf-8 -*-
"""
Execution State Management for Unified Communication.

Provides thread-safe execution state management for safely sharing execution
state and feedback data between the HTTP server running in a separate thread
and the main thread.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ExecutionState:
    """Execution state data model.
    
    Stores the complete state information for a single skill list execution.
    
    Attributes:
        execution_id: Unique execution identifier
        status: Execution status ("pending", "running", "completed", "failed", "interrupted")
        completed: Whether execution completed successfully
        interrupted: Whether execution was interrupted
        total_timesteps: Total number of timesteps
        completed_timesteps: Number of completed timesteps
        feedbacks: List of all received feedbacks
        message: Completion message
        start_time: Execution start timestamp
        end_time: Execution end timestamp (optional)
    """
    execution_id: str
    status: str = "pending"
    completed: bool = False
    interrupted: bool = False
    total_timesteps: int = 0
    completed_timesteps: int = 0
    feedbacks: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


class ThreadSafeExecutionState:
    """Thread-safe execution state manager.
    
    Uses threading.Lock to protect state reads/writes and threading.Event
    to provide cross-thread completion signal notification.
    
    Example:
        state = ThreadSafeExecutionState()
        
        # Set a new execution state
        execution = ExecutionState(execution_id="abc-123")
        state.set_execution(execution)
        
        # Add feedback from another thread
        state.add_feedback({"feedback_type": "timestep", "time_step": 0})
        
        # Mark as complete
        state.mark_complete(completed=True, interrupted=False)
        
        # Main thread waits for completion
        if state.wait_for_complete(timeout=30.0):
            result = state.get_execution()
    """
    
    def __init__(self):
        """Initialize the thread-safe state manager."""
        self._lock = threading.Lock()
        self._execution: Optional[ExecutionState] = None
        self._complete_event = threading.Event()
    
    def set_execution(self, execution: ExecutionState) -> None:
        """Set the current execution state.
        
        Sets a new execution state and clears the completion event,
        preparing to receive new execution feedback.
        
        Args:
            execution: The new execution state object
        """
        with self._lock:
            self._execution = execution
            self._complete_event.clear()
    
    def get_execution(self) -> Optional[ExecutionState]:
        """Get the current execution state.
        
        Returns:
            A shallow copy of the current execution state, or None if no state exists
        """
        with self._lock:
            if self._execution is None:
                return None
            # Return a shallow copy to prevent external modifications from affecting internal state
            return ExecutionState(
                execution_id=self._execution.execution_id,
                status=self._execution.status,
                completed=self._execution.completed,
                interrupted=self._execution.interrupted,
                total_timesteps=self._execution.total_timesteps,
                completed_timesteps=self._execution.completed_timesteps,
                feedbacks=self._execution.feedbacks.copy(),
                message=self._execution.message,
                start_time=self._execution.start_time,
                end_time=self._execution.end_time
            )
    
    def mark_complete(
        self, 
        completed: bool, 
        interrupted: bool,
        total_timesteps: int = 0,
        completed_timesteps: int = 0,
        message: str = ""
    ) -> None:
        """Mark execution as complete.
        
        Updates the execution state to completed/interrupted/failed and
        triggers the completion event to notify waiting threads.
        
        Args:
            completed: Whether execution completed successfully
            interrupted: Whether execution was interrupted
            total_timesteps: Total number of timesteps
            completed_timesteps: Number of completed timesteps
            message: Completion message
        """
        with self._lock:
            if self._execution:
                self._execution.completed = completed
                self._execution.interrupted = interrupted
                self._execution.total_timesteps = total_timesteps
                self._execution.completed_timesteps = completed_timesteps
                self._execution.message = message
                self._execution.end_time = time.time()
                
                if completed:
                    self._execution.status = "completed"
                elif interrupted:
                    self._execution.status = "interrupted"
                else:
                    self._execution.status = "failed"
        
        # Set event outside the lock to avoid deadlock
        self._complete_event.set()
    
    def wait_for_complete(self, timeout: float) -> bool:
        """Wait for execution to complete.
        
        Blocks the current thread until execution completes or times out.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if completed before timeout, False if timed out
        """
        return self._complete_event.wait(timeout=timeout)
    
    def add_feedback(self, feedback: Dict[str, Any]) -> None:
        """Add feedback.
        
        Thread-safely appends feedback to the current execution state's feedback list.
        
        Args:
            feedback: Feedback data dictionary
        """
        with self._lock:
            if self._execution:
                self._execution.feedbacks.append(feedback)
    
    def update_timestep_progress(self, completed_timesteps: int) -> None:
        """Update timestep progress.
        
        Args:
            completed_timesteps: Number of completed timesteps
        """
        with self._lock:
            if self._execution:
                self._execution.completed_timesteps = completed_timesteps
                if self._execution.status == "pending":
                    self._execution.status = "running"
    
    def clear(self) -> None:
        """Clear the current execution state.
        
        Resets all state, preparing to receive a new execution.
        """
        with self._lock:
            self._execution = None
        self._complete_event.clear()
    
    def is_complete(self) -> bool:
        """Check whether execution has completed.
        
        Returns:
            True if the completion event has been set
        """
        return self._complete_event.is_set()
    
    def get_feedbacks(self) -> List[Dict[str, Any]]:
        """Get a copy of all feedbacks.
        
        Returns:
            A copy of the feedback list
        """
        with self._lock:
            if self._execution:
                return self._execution.feedbacks.copy()
            return []
