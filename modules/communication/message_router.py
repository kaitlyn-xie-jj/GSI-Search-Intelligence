"""
Message Router for HITL communication system.

This module provides the MessageRouter class that routes incoming messages
to appropriate handlers based on their message category.
"""

import logging
from typing import Any, Callable, Dict, List

from modules.communication.enums import MessageCategory, MessageDirection
from modules.communication.message import HITLMessage


logger = logging.getLogger(__name__)


class MessageRouter:
    """
    Message Router - Routes messages to appropriate handlers based on category.
    
    The MessageRouter maintains a registry of handlers for each message category
    and dispatches incoming messages to the registered handlers.
    
    Attributes:
        _handlers: Dictionary mapping MessageCategory to list of handler functions
    """
    
    def __init__(self):
        """Initialize the message router with empty handler registries."""
        self._handlers: Dict[MessageCategory, List[Callable[[HITLMessage], None]]] = {
            cat: [] for cat in MessageCategory
        }
    
    def register(
        self,
        category: MessageCategory,
        handler: Callable[[HITLMessage], None]
    ) -> None:
        """
        Register a message handler for a specific category.
        
        Args:
            category: The message category to handle
            handler: Callable that accepts an HITLMessage and processes it
        """
        if category not in self._handlers:
            self._handlers[category] = []
        self._handlers[category].append(handler)
        logger.debug(f"Registered handler for category {category.value}")
    
    def unregister(
        self,
        category: MessageCategory,
        handler: Callable[[HITLMessage], None]
    ) -> bool:
        """
        Unregister a message handler for a specific category.
        
        Args:
            category: The message category
            handler: The handler to remove
            
        Returns:
            True if handler was found and removed, False otherwise
        """
        if category in self._handlers and handler in self._handlers[category]:
            self._handlers[category].remove(handler)
            logger.debug(f"Unregistered handler for category {category.value}")
            return True
        return False
    
    def dispatch(self, message: HITLMessage) -> None:
        """
        Dispatch a message to all registered handlers for its category.
        
        Args:
            message: The HITLMessage to dispatch
        """
        handlers = self._handlers.get(message.message_category, [])
        if not handlers:
            logger.warning(
                f"No handlers registered for category {message.message_category.value}, "
                f"message_id={message.message_id}"
            )
            return
        
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(
                    f"Handler error for category {message.message_category.value}, "
                    f"message_id={message.message_id}: {e}",
                    exc_info=True
                )
    
    def route_inbound(self, raw_message: Dict[str, Any]) -> None:
        """
        Route an inbound raw message dictionary to appropriate handlers.
        
        This method parses the raw message dictionary, validates its format,
        creates an HITLMessage instance, and dispatches it to handlers.
        
        Args:
            raw_message: Dictionary containing the raw message data
            
        Note:
            Invalid messages are logged and discarded without raising exceptions.
        """
        try:
            # Validate required fields
            required_fields = [
                "message_id", "message_category", "message_type",
                "direction", "timestamp"
            ]
            missing_fields = [f for f in required_fields if f not in raw_message]
            if missing_fields:
                logger.error(
                    f"Invalid message format: missing fields {missing_fields}"
                )
                return
            
            # Parse message category
            try:
                category = MessageCategory(raw_message["message_category"])
            except ValueError:
                logger.error(
                    f"Invalid message_category: {raw_message.get('message_category')}"
                )
                return
            
            # Parse direction
            try:
                direction = MessageDirection(raw_message["direction"])
            except ValueError:
                logger.error(
                    f"Invalid direction: {raw_message.get('direction')}"
                )
                return
            
            # Create HITLMessage instance
            message = HITLMessage(
                message_id=raw_message["message_id"],
                message_category=category,
                message_type=raw_message["message_type"],
                direction=direction,
                timestamp=raw_message["timestamp"],
                payload=raw_message.get("payload", {})
            )
            
            logger.debug(
                f"Routing inbound message: category={category.value}, "
                f"type={message.message_type}, id={message.message_id}"
            )
            
            # Dispatch to handlers
            self.dispatch(message)
            
        except Exception as e:
            logger.error(f"Error routing inbound message: {e}", exc_info=True)
