# -*- coding: utf-8 -*-
"""
Unified Communicator for Python-UE5 Communication.

Provides all HTTP communication between Python and UE5, including:
- Platform messages (skill lists, task feedback, world state, health status)
- HITL messages (instructions, plan review, result decisions)

Architecture notes:
- Python acts as server: provides /api/sim/poll, /api/hitl/poll for UE5 to poll
- Python acts as client: queries UE5's /api/world_state, /api/health
- Uses a separate thread to run the HTTP server, avoiding blocking the main asyncio event loop
"""

import asyncio
import logging
import queue
import threading
import uuid
from datetime import datetime
from http.server import HTTPServer
from typing import Any, Callable, Dict, List, Optional

# HTTP client - use aiohttp for async client requests
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from modules.communication.enums import MessageCategory, MessageDirection
from modules.communication.message import HITLMessage
from modules.communication.message_router import MessageRouter
from modules.communication.execution_state import (
    ExecutionState,
    ThreadSafeExecutionState,
)
from modules.communication.unified_http_handler import UnifiedHttpRequestHandler


logger = logging.getLogger(__name__)


class UnifiedCommunicator:
    """Unified Communicator - Handles all HTTP communication between Python and UE5.
    
    Attributes:
        unreal_url: URL of the UE5 API
        server_port: Python HTTP server port
        timeout: Request timeout duration
        hitl_enabled: Whether HITL mode is enabled
    """
    
    def __init__(
        self,
        unreal_url: str = "http://localhost:8080",
        server_port: int = 8081,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        hitl_enabled: bool = False,
        logger_instance: Optional[logging.Logger] = None
    ):
        """Initialize the unified communicator.
        
        Args:
            unreal_url: URL of the UE5 API
            server_port: Python HTTP server port
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            hitl_enabled: Whether to enable HITL mode
            logger_instance: Custom logger instance
        """
        self.unreal_url = unreal_url.rstrip('/')
        self.server_port = server_port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._hitl_enabled = hitl_enabled
        
        # Logger
        self._logger = logger_instance or logging.getLogger(__name__)
        
        # HTTP client session (for accessing UE5)
        self._session: Optional[Any] = None  # aiohttp.ClientSession
        
        # ========== Platform message queues ==========
        # Outbound queue (skill lists)
        self._platform_outbound_queue: queue.Queue = queue.Queue()
        # Feedback queue (for incremental processing)
        self._platform_feedback_queue: queue.Queue = queue.Queue()
        # Execution state manager
        self._execution_state = ThreadSafeExecutionState()
        
        # ========== HITL message queues ==========
        # Outbound queues (by category)
        self._hitl_outbound_queues: Dict[MessageCategory, queue.Queue] = {
            cat: queue.Queue() for cat in MessageCategory
        }
        # Inbound queues (by category)
        self._hitl_inbound_queues: Dict[MessageCategory, queue.Queue] = {
            cat: queue.Queue() for cat in MessageCategory
        }
        # Message router
        self._router = MessageRouter()
        
        # ========== HTTP server ==========
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_running = False
        self._server_lock = threading.Lock()
        
        # Log initialization info
        self._logger.info(
            f"UnifiedCommunicator initialized: "
            f"unreal_url={unreal_url}, server_port={server_port}, "
            f"hitl_enabled={hitl_enabled}"
        )

    # ========== HTTP Server ==========
    
    async def start_server(self) -> None:
        """Start the HTTP server (idempotent operation).
        
        Starts the HTTP server in a separate thread. If the server is
        already running, it will not be started again.
        """
        with self._server_lock:
            if self._server_running:
                self._logger.info(f"Server already running on port {self.server_port}")
                return
            
            # Configure shared state for the HTTP handler
            UnifiedHttpRequestHandler.platform_outbound_queue = self._platform_outbound_queue
            UnifiedHttpRequestHandler.platform_feedback_queue = self._platform_feedback_queue
            UnifiedHttpRequestHandler.execution_state = self._execution_state
            UnifiedHttpRequestHandler.hitl_outbound_queues = self._hitl_outbound_queues
            UnifiedHttpRequestHandler.hitl_inbound_queues = self._hitl_inbound_queues
            UnifiedHttpRequestHandler.message_router = self._router
            UnifiedHttpRequestHandler.hitl_enabled = self._hitl_enabled
            UnifiedHttpRequestHandler.custom_logger = self._logger
            
            # Create the HTTP server
            try:
                self._server = HTTPServer(
                    ('0.0.0.0', self.server_port),
                    UnifiedHttpRequestHandler
                )
            except OSError as e:
                self._logger.error(f"Failed to bind to port {self.server_port}: {e}")
                raise
            
            # Start the server in a separate thread
            self._server_thread = threading.Thread(
                target=self._run_server,
                name="UnifiedCommunicatorServer",
                daemon=True
            )
            self._server_thread.start()
            self._server_running = True
        
        self._logger.info(f"HTTP server started on port {self.server_port}")
        self._logger.info(f"  - GET  /api/sim/poll    (UE5 polls for skill list)")
        self._logger.info(f"  - POST /api/sim/message (UE5 sends feedback)")
        if self._hitl_enabled:
            self._logger.info(f"  - GET  /api/hitl/poll   (UE5 polls for HITL messages)")
            self._logger.info(f"  - POST /api/hitl/message (UE5 sends HITL messages)")
        self._logger.info(f"  - GET  /api/health      (Health check)")
    
    def _run_server(self) -> None:
        """Run the HTTP server in a separate thread."""
        try:
            self._server.serve_forever()
        except Exception as e:
            self._logger.error(f"HTTP server error: {e}")
        finally:
            self._server_running = False
    
    async def stop_server(self) -> None:
        """Stop the HTTP server.
        
        Gracefully shuts down the server and cleans up resources.
        """
        with self._server_lock:
            if not self._server_running:
                return
            
            if self._server:
                self._server.shutdown()
                self._server.server_close()
                self._server = None
            
            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join(timeout=5.0)
            self._server_thread = None
            
            self._server_running = False
            
            # Clean up platform queues
            while not self._platform_outbound_queue.empty():
                try:
                    self._platform_outbound_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Clean up HITL queues
            for cat in MessageCategory:
                while not self._hitl_outbound_queues[cat].empty():
                    try:
                        self._hitl_outbound_queues[cat].get_nowait()
                    except queue.Empty:
                        break
                while not self._hitl_inbound_queues[cat].empty():
                    try:
                        self._hitl_inbound_queues[cat].get_nowait()
                    except queue.Empty:
                        break
            
            # Clear execution state
            self._execution_state.clear()
        
        self._logger.info("HTTP server stopped")

    # ========== HTTP Client (accessing UE5) ==========
    
    async def _get_session(self) -> Any:
        """Get or create an HTTP session."""
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp is required for HTTP client functionality")
        
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def _request_with_retry(self, method: str, url: str, **kwargs) -> Any:
        """HTTP request with retry mechanism."""
        session = await self._get_session()
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = await session.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = await session.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                return response
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                self._logger.warning(
                    f"HTTP request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        raise aiohttp.ClientError(f"All {self.max_retries} attempts failed: {last_error}")

    # ========== Platform Message Interface ==========
    
    async def send_skill_list(self, skills: Dict[str, Any]) -> Dict[str, Any]:
        """Send a skill list to the queue for UE5 to poll.
        
        Args:
            skills: Skill list in the format {timestep: {robot: {skill, params}}}
            
        Returns:
            Response containing the execution_id
        """
        execution_id = str(uuid.uuid4())
        
        # Build the message
        message = {
            "message_type": "skill_list",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "message_id": execution_id,
            "payload": skills
        }
        
        # Calculate total timesteps
        total_timesteps = len(skills) if isinstance(skills, dict) else 0
        
        # Set a new execution state
        execution = ExecutionState(
            execution_id=execution_id,
            status="pending",
            total_timesteps=total_timesteps
        )
        self._execution_state.set_execution(execution)
        
        # Enqueue the message
        self._platform_outbound_queue.put(message)
        
        self._logger.info(
            f"Skill list queued (execution_id: {execution_id[:8]}..., "
            f"queue_size: {self._platform_outbound_queue.qsize()})"
        )
        return {"execution_id": execution_id, "status": "queued"}
    
    async def wait_for_completion(
        self,
        execution_id: str,
        max_wait: float = 1000.0,
        on_feedback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Wait for execution to complete, with incremental feedback processing.
        
        Args:
            execution_id: Execution ID
            max_wait: Maximum wait time in seconds
            on_feedback: Incremental feedback callback, invoked for each timestep feedback.
                        Signature: async def callback(feedback: Dict) -> None
            
        Returns:
            Final execution result
            
        Raises:
            TimeoutError: If the wait times out
        """
        self._logger.info(f"Waiting for execution {execution_id} to complete (max {max_wait}s)")
        
        import time
        start_time = time.time()
        poll_interval = 0.05  # 50ms polling interval
        processed_timesteps = set()
        
        while True:
            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                raise TimeoutError(f"Execution {execution_id} timed out after {max_wait}s")
            
            # Process all feedbacks in the feedback queue (incremental processing)
            while not self._platform_feedback_queue.empty():
                try:
                    feedback = self._platform_feedback_queue.get_nowait()
                    feedback_type = feedback.get("feedback_type")
                    
                    if feedback_type == "timestep":
                        time_step = feedback.get("time_step", -1)
                        processed_timesteps.add(time_step)
                        if on_feedback:
                            try:
                                await on_feedback(feedback)
                            except Exception as e:
                                self._logger.error(f"Error in feedback callback: {e}")
                except queue.Empty:
                    break
            
            # Check if complete
            if self._execution_state.is_complete():
                break
            
            # Brief wait
            await asyncio.sleep(poll_interval)
        
        # After completion, wait briefly to ensure all feedbacks have been enqueued
        await asyncio.sleep(0.1)
        
        # Process remaining feedbacks
        while not self._platform_feedback_queue.empty():
            try:
                feedback = self._platform_feedback_queue.get_nowait()
                feedback_type = feedback.get("feedback_type")
                
                if feedback_type == "timestep":
                    time_step = feedback.get("time_step", -1)
                    processed_timesteps.add(time_step)
                    if on_feedback:
                        try:
                            await on_feedback(feedback)
                        except Exception as e:
                            self._logger.error(f"Error in feedback callback: {e}")
            except queue.Empty:
                break
        
        # Collect results
        execution = self._execution_state.get_execution()
        
        if execution:
            return {
                "execution_id": execution_id,
                "status": execution.status,
                "success": execution.completed,
                "interrupted": execution.interrupted,
                "total_timesteps": execution.total_timesteps,
                "completed_timesteps": execution.completed_timesteps,
                "feedbacks": execution.feedbacks
            }
        else:
            return {
                "execution_id": execution_id,
                "status": "unknown",
                "success": False,
                "feedbacks": []
            }
    
    async def get_world_state(self) -> Dict[str, Any]:
        """Get the UE5 world state."""
        url = f"{self.unreal_url}/api/world_state"
        self._logger.debug(f"Getting world state from {url}")
        
        async with await self._request_with_retry("GET", url) as resp:
            resp.raise_for_status()
            data = await resp.json()
            self._logger.info(
                f"Retrieved world state with "
                f"{len(data.get('nodes', data.get('entities', [])))} entities"
            )
            return data
    
    async def health_check(self) -> bool:
        """Check whether the UE5 API is available."""
        try:
            url = f"{self.unreal_url}/api/health"
            async with await self._request_with_retry("GET", url) as resp:
                return resp.status == 200
        except Exception as e:
            self._logger.warning(f"Health check failed: {e}")
            return False

    # ========== HITL Message Interface ==========
    
    def send_hitl_message(self, message: HITLMessage) -> None:
        """Send an HITL message to the outbound queue.
        
        Args:
            message: The HITL message to send
        """
        
        if not self._hitl_enabled:
            self._logger.warning("HITL not enabled, message not sent")
            return
        
        q = self._hitl_outbound_queues[message.message_category]
        q.put(message)
        
        self._logger.info(
            f"HITL message queued for sending: "
            f"category={message.message_category.value}, "
            f"type={message.message_type}, id={message.message_id}, "
            f"queue_size={q.qsize()}"
        )
    
    async def wait_for_hitl_response(
        self,
        category: MessageCategory,
        timeout: Optional[float] = None
    ) -> Optional[HITLMessage]:
        """Wait for an HITL response.
        
        Args:
            category: Message category
            timeout: Timeout in seconds, defaults to self.timeout
            
        Returns:
            The received HITL message, or None on timeout
        """
        if not self._hitl_enabled:
            self._logger.warning("HITL not enabled")
            return None
        
        timeout = timeout if timeout is not None else self.timeout
        import time
        start_time = time.time()
        poll_interval = 0.05
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self._logger.warning(
                    f"Timeout waiting for HITL response: "
                    f"category={category.value}, timeout={timeout}s"
                )
                return None
            
            try:
                message = self._hitl_inbound_queues[category].get_nowait()
                self._logger.debug(
                    f"HITL response received: category={category.value}, "
                    f"type={message.message_type}, id={message.message_id}"
                )
                return message
            except queue.Empty:
                await asyncio.sleep(poll_interval)
    
    def register_hitl_handler(
        self,
        category: MessageCategory,
        handler: Callable[[HITLMessage], None]
    ) -> None:
        """Register an HITL message handler.
        
        Args:
            category: Message category
            handler: Handler function
        """
        self._router.register(category, handler)
        self._logger.debug(f"HITL handler registered for category: {category.value}")

    # ========== Queue Access ==========
    
    def get_outbound_queue(self, category: MessageCategory) -> queue.Queue:
        """Get the outbound queue for the specified category.
        
        Args:
            category: Message category
            
        Returns:
            The corresponding outbound queue
        """
        if category == MessageCategory.PLATFORM:
            return self._platform_outbound_queue
        return self._hitl_outbound_queues[category]
    
    def get_inbound_queue(self, category: MessageCategory) -> queue.Queue:
        """Get the inbound queue for the specified category.
        
        Args:
            category: Message category
            
        Returns:
            The corresponding inbound queue
        """
        return self._hitl_inbound_queues[category]

    # ========== Lifecycle Management ==========
    
    async def close(self) -> None:
        """Close all connections."""
        await self.stop_server()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._logger.debug("UnifiedCommunicator closed")
    
    @property
    def is_server_running(self) -> bool:
        """Check whether the server is running."""
        return self._server_running
    
    @property
    def hitl_enabled(self) -> bool:
        """Check whether HITL is enabled."""
        return self._hitl_enabled
    
    @hitl_enabled.setter
    def hitl_enabled(self, value: bool) -> None:
        """Set the HITL enabled state.
        
        Note: If the server is already running, a restart is required
        for the change to take effect.
        """
        self._hitl_enabled = value
        # Update the handler's configuration
        UnifiedHttpRequestHandler.hitl_enabled = value
        self._logger.info(f"HITL enabled set to: {value}")
    
    async def __aenter__(self) -> 'UnifiedCommunicator':
        await self.start_server()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    def __repr__(self) -> str:
        return (
            f"UnifiedCommunicator("
            f"unreal_url={self.unreal_url}, "
            f"server_port={self.server_port}, "
            f"hitl_enabled={self._hitl_enabled})"
        )
