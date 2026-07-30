"""
HITL Message data class for standardized message format.

This module defines the HITLMessage dataclass that provides a unified
message format for all HITL communications between Python and UE5.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

from modules.communication.enums import MessageCategory, MessageDirection


@dataclass
class HITLMessage:
    """
    HITL Message data structure.
    
    Represents a standardized message format for all human-in-the-loop
    communications. Each message contains:
    - message_id: Unique identifier (UUID)
    - message_category: Category of the message (INSTRUCTION, REVIEW, etc.)
    - message_type: Specific operation type within the category
    - direction: Direction of message flow
    - timestamp: ISO 8601 formatted timestamp
    - payload: Category-specific data dictionary
    
    Attributes:
        message_id: Unique identifier for the message (UUID string)
        message_category: The category of this message
        message_type: The specific type/operation within the category
        direction: Direction of message transmission
        timestamp: ISO 8601 formatted timestamp string
        payload: Dictionary containing message-specific data
    """
    message_id: str
    message_category: MessageCategory
    message_type: str
    direction: MessageDirection
    timestamp: str
    payload: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        category: MessageCategory,
        msg_type: str,
        direction: MessageDirection,
        payload: Optional[Dict[str, Any]] = None
    ) -> 'HITLMessage':
        """
        Factory method to create a new HITLMessage with auto-generated ID and timestamp.
        
        Args:
            category: The message category
            msg_type: The specific message type within the category
            direction: The direction of message flow
            payload: Optional dictionary containing message-specific data
            
        Returns:
            A new HITLMessage instance with generated UUID and ISO 8601 timestamp
        """
        return cls(
            message_id=str(uuid.uuid4()),
            message_category=category,
            message_type=msg_type,
            direction=direction,
            timestamp=datetime.now().isoformat(),
            payload=payload if payload is not None else {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the message to a dictionary format.
        
        Returns:
            Dictionary representation of the message with enum values
            converted to their string representations.
        """
        return {
            "message_id": self.message_id,
            "message_category": self.message_category.value,
            "message_type": self.message_type,
            "direction": self.direction.value,
            "timestamp": self.timestamp,
            "payload": self.payload
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HITLMessage':
        """
        Deserialize a dictionary to create an HITLMessage instance.
        
        Args:
            data: Dictionary containing message fields. The message_category
                  and direction fields can be either enum values or strings.
                  
        Returns:
            A new HITLMessage instance populated from the dictionary data.
            
        Raises:
            KeyError: If required fields are missing from the dictionary
            ValueError: If enum values are invalid
        """
        # Handle message_category - can be string or enum
        category = data["message_category"]
        if isinstance(category, str):
            category = MessageCategory(category)
        
        # Handle direction - can be string or enum
        direction = data["direction"]
        if isinstance(direction, str):
            direction = MessageDirection(direction)
        
        return cls(
            message_id=data["message_id"],
            message_category=category,
            message_type=data["message_type"],
            direction=direction,
            timestamp=data["timestamp"],
            payload=data.get("payload", {})
        )
