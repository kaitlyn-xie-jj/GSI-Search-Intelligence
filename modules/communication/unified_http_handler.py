# -*- coding: utf-8 -*-
"""
Unified HTTP Request Handler for Communication System.

Endpoints:
    GET  /api/sim/poll      - UE5 polls for skill lists
    POST /api/sim/message   - UE5 sends execution feedback
    GET  /api/hitl/poll     - UE5 polls for HITL messages
    POST /api/hitl/message  - UE5 sends HITL responses
    GET  /api/health        - Health check
"""

import json
import logging
import queue
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional

from modules.communication.enums import MessageCategory, MessageDirection
from modules.communication.message import HITLMessage
from modules.communication.execution_state import ThreadSafeExecutionState


logger = logging.getLogger(__name__)


class UnifiedHttpRequestHandler(BaseHTTPRequestHandler):
    """Unified HTTP Request Handler.
    
    Handles all platform and HITL communication endpoints, with HITL
    functionality controlled by the hitl_enabled flag.
    
    Endpoints:
        GET  /api/sim/poll      - UE5 polls for skill lists
        POST /api/sim/message   - UE5 sends execution feedback
        GET  /api/hitl/poll     - UE5 polls for HITL messages (requires HITL enabled)
        POST /api/hitl/message  - UE5 sends HITL responses (requires HITL enabled)
        GET  /api/health        - Health check
    
    Class Attributes:
        platform_outbound_queue: Platform outbound message queue (skill lists)
        platform_feedback_queue: Platform feedback queue (for incremental processing)
        execution_state: Execution state manager
        hitl_outbound_queues: HITL outbound queues (by category)
        hitl_inbound_queues: HITL inbound queues (by category)
        message_router: Message router
        hitl_enabled: Whether HITL is enabled
        custom_logger: Logger instance
    """
    
    # Platform communication (skill lists, feedback)
    platform_outbound_queue: queue.Queue = None
    platform_feedback_queue: queue.Queue = None
    execution_state: ThreadSafeExecutionState = None
    
    # HITL communication
    hitl_outbound_queues: Dict[MessageCategory, queue.Queue] = None
    hitl_inbound_queues: Dict[MessageCategory, queue.Queue] = None
    message_router: Any = None  # MessageRouter
    
    # Configuration
    hitl_enabled: bool = False
    custom_logger: logging.Logger = None

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/api/sim/poll':
            self._handle_sim_poll()
        elif self.path == '/api/hitl/poll':
            self._handle_hitl_poll()
        elif self.path == '/api/health':
            self._handle_health()
        else:
            self._send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/api/sim/message':
            self._handle_sim_message()
        elif self.path == '/api/hitl/message':
            self._handle_hitl_message()
        else:
            self._send_error(404, "Not Found")

    # ========== Platform Endpoints ==========
    
    def _handle_sim_poll(self):
        """Handle GET /api/sim/poll - Platform skill list polling.
        
        Retrieves pending skill lists from the message queue:
        - Returns HTTP 200 with message content when a message is available
        - Returns HTTP 204 when no messages are available
        """
        if self.platform_outbound_queue is None:
            self._send_error(500, "Server not properly initialized")
            return
        
        try:
            msg = self.platform_outbound_queue.get_nowait()
            msg_id = msg.get('message_id', 'unknown')[:8]
            self._log(f">>> Sending skill_list to UE5 (message_id: {msg_id}...)")
            
            try:
                self._send_json_response(200, [msg])
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                # Send failed, return message to queue
                self.platform_outbound_queue.put(msg)
                self._log(f"Client disconnected, message returned to queue: {e}", warning=True)
                
        except queue.Empty:
            self._send_204()
    
    def _handle_sim_message(self):
        """Handle POST /api/sim/message - Platform execution feedback.
        
        Receives execution feedback sent by UE5.
        """
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error(400, "Empty request body")
                return
            
            body = self.rfile.read(content_length).decode('utf-8')
            
            try:
                message = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_error(400, f"Invalid JSON: {e}")
                return
            
            self._process_sim_message(message)
            self._send_json_response(200, {"status": "ok"})
            
        except Exception as e:
            self._log(f"Error handling sim message: {e}", error=True)
            self._send_error(500, f"Internal error: {e}")
    
    def _process_sim_message(self, message: Dict[str, Any]):
        """Process a platform message."""
        msg_type = message.get('message_type', '')
        payload = message.get('payload', {})
        feedback_type = payload.get('feedback_type', '')
        
        self._log(f"<<< Received sim message: type={msg_type}, feedback_type={feedback_type}")
        
        if msg_type != 'task_feedback':
            self._log(f"Received non-feedback message: {msg_type}")
            return
        
        if self.execution_state is None:
            self._log("Execution state not initialized, ignoring feedback", warning=True)
            return
        
        # Store feedback in state
        self.execution_state.add_feedback(payload)
        
        # Enqueue feedback for incremental processing
        if self.platform_feedback_queue is not None:
            self.platform_feedback_queue.put(payload)
        
        if feedback_type == 'skill_list_completed':
            self._handle_sim_completion(payload)
        elif feedback_type == 'timestep':
            time_step = payload.get('time_step', -1)
            self.execution_state.update_timestep_progress(time_step + 1)
            self._log(f"<<< TimeStep {time_step} feedback received and processed")
    
    def _handle_sim_completion(self, payload: Dict[str, Any]):
        """Handle skill list completion feedback."""
        completed = payload.get('completed', False)
        interrupted = payload.get('interrupted', False)
        total = payload.get('total_time_steps', 0)
        done = payload.get('completed_time_steps', 0)
        
        status = "COMPLETED" if completed else ("INTERRUPTED" if interrupted else "FAILED")
        self._log(f"<<< Skill list {status}: {done}/{total} steps")
        
        self.execution_state.mark_complete(
            completed=completed,
            interrupted=interrupted,
            total_timesteps=total,
            completed_timesteps=done,
            message=payload.get('message', '')
        )

    # ========== HITL Endpoints ==========
    
    def _handle_hitl_poll(self):
        """Handle GET /api/hitl/poll - HITL message polling.
        
        Returns pending HITL messages for UE5 to process.
        Messages are returned in priority order: REVIEW, DECISION, INSTRUCTION.
        
        When HITL is not enabled, returns 204 No Content (instead of an error),
        so the UE5 side can poll normally without receiving errors.
        """
        
        # When HITL is not enabled, return 204 No Content (silent handling)
        if not self.hitl_enabled:
            self._send_204()
            return
        
        if self.hitl_outbound_queues is None:
            self._log("HITL poll: hitl_outbound_queues is None", error=True)
            self._send_error(500, "HITL not properly initialized")
            return
        
        # Check queues in priority order
        priority_order = [
            MessageCategory.REVIEW,
            MessageCategory.DECISION,
            MessageCategory.INSTRUCTION,
        ]
        
        # Debug: log queue sizes
        queue_info = {}
        for cat in priority_order:
            q = self.hitl_outbound_queues.get(cat)
            if q:
                queue_info[cat.value] = {"size": q.qsize(), "id": id(q)}
        
        total_size = sum(info["size"] for info in queue_info.values())
        
        if total_size > 0:
            self._log(f"HITL poll: found {total_size} messages, queue_info={queue_info}")
        
        messages = []
        for category in priority_order:
            q = self.hitl_outbound_queues.get(category)
            if q is not None:
                try:
                    msg = q.get_nowait()
                    
                    if isinstance(msg, HITLMessage):
                        msg_dict = msg.to_dict()
                        messages.append(msg_dict)
                    else:
                        messages.append(msg)
                    
                    self._log(
                        f">>> Sending HITL message to UE5: "
                        f"category={category.value}"
                    )
                except queue.Empty:
                    continue
        
        if messages:
            self._send_json_response(200, messages)
        else:
            self._send_204()
    
    def _handle_hitl_message(self):
        """Handle POST /api/hitl/message - HITL message reception.
        
        Receives HITL messages from UE5 (instructions, review responses, decisions)
        and routes them to the appropriate handlers.
        
        When HITL is not enabled, returns 200 OK but does not process the message
        (silently discarded), so the UE5 side can send normally without receiving errors.
        """
        # When HITL is not enabled, silently accept but do not process
        if not self.hitl_enabled:
            self._log("HITL message received but HITL not enabled, ignoring", warning=True)
            self._send_json_response(200, {"status": "ok", "note": "HITL not enabled, message ignored"})
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error(400, "Empty request body")
                return
            
            body = self.rfile.read(content_length).decode('utf-8')
            
            try:
                raw_message = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_error(400, f"Invalid JSON: {e}")
                return
            
            self._process_hitl_message(raw_message)
            self._send_json_response(200, {"status": "ok"})
            
        except Exception as e:
            self._log(f"Error handling HITL message: {e}", error=True)
            self._send_error(500, f"Internal error: {e}")
    
    def _process_hitl_message(self, raw_message: Dict[str, Any]):
        """Process an HITL message.
        
        Validates the message format, creates an HITLMessage instance,
        enqueues it in the corresponding inbound queue, and routes it to handlers.
        """
        self._log(f"<<< Received HITL message: {raw_message.get('message_category')}")
        
        # Validate required fields
        required_fields = [
            "message_id", "message_category", "message_type",
            "direction", "timestamp"
        ]
        missing_fields = [f for f in required_fields if f not in raw_message]
        if missing_fields:
            self._log(
                f"Invalid HITL message: missing fields {missing_fields}",
                error=True
            )
            return
        
        try:
            # Parse and create HITLMessage
            message = HITLMessage.from_dict(raw_message)
            
            # Enqueue in the inbound queue
            if self.hitl_inbound_queues is not None:
                q = self.hitl_inbound_queues.get(message.message_category)
                if q is not None:
                    q.put(message)
                    self._log(
                        f"<<< HITL message queued: "
                        f"category={message.message_category.value}, "
                        f"type={message.message_type}"
                    )
            
            # Route to handlers
            if self.message_router is not None:
                self.message_router.dispatch(message)
                
        except (ValueError, KeyError) as e:
            self._log(f"Error parsing HITL message: {e}", error=True)
    
    def _handle_health(self):
        """Handle GET /api/health - Health check endpoint."""
        response = {
            "status": "healthy",
            "hitl_enabled": self.hitl_enabled
        }
        self._send_json_response(200, response)

    # ========== Helper Methods ==========
    
    def _send_json_response(self, status_code: int, data: Any):
        """Send a JSON response."""
        try:
            body = json.dumps(data).encode('utf-8')
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            pass  # Client already disconnected
    
    def _send_204(self):
        """Send a 204 No Content response."""
        try:
            self.send_response(204)
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
    
    def _send_error(self, status_code: int, message: str):
        """Send an error response."""
        self._log(f"HTTP {status_code}: {message}", warning=True)
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(message.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
    
    def _log(self, msg: str, warning: bool = False, error: bool = False):
        """Unified log output."""
        if self.custom_logger:
            if error:
                self.custom_logger.error(msg)
            elif warning:
                self.custom_logger.warning(msg)
            else:
                self.custom_logger.info(msg)
        elif logger:
            if error:
                logger.error(msg)
            elif warning:
                logger.warning(msg)
            else:
                logger.debug(msg)
    
    def log_message(self, format: str, *args):
        """Override default log method to suppress stderr output."""
        pass
