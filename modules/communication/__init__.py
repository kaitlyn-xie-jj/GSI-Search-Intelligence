"""
Communication module for Human-in-the-Loop (HITL) interaction system.

This module provides a unified bidirectional communication layer that supports
multiple message categories including INSTRUCTION, REVIEW, DECISION, and PLATFORM.

Unified communication components:
- UnifiedCommunicator: Unified communicator integrating platform and HITL communication (recommended)
- UnifiedHttpRequestHandler: Unified HTTP request handler (recommended)
- ExecutionState: Execution state data model
- ThreadSafeExecutionState: Thread-safe execution state manager
"""

from modules.communication.enums import (
    MessageCategory,
    MessageDirection,
    ReviewType,
    DecisionType,
)
from modules.communication.message import HITLMessage
from modules.communication.message_router import MessageRouter
from modules.communication.execution_state import (
    ExecutionState,
    ThreadSafeExecutionState,
)
from modules.communication.unified_http_handler import UnifiedHttpRequestHandler
from modules.communication.unified_communicator import UnifiedCommunicator

__all__ = [
    # Enum types
    "MessageCategory",
    "MessageDirection",
    "ReviewType",
    "DecisionType",
    # Message class
    "HITLMessage",
    # Router
    "MessageRouter",
    # Others
    "UnifiedCommunicator",
    "UnifiedHttpRequestHandler",
    "ExecutionState",
    "ThreadSafeExecutionState",
]
