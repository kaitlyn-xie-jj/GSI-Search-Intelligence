"""
HITL Configuration and State Data Classes.

This module defines the configuration and state dataclasses for the
Human-in-the-Loop interaction system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HITLConfig:
    """
    HITL Configuration dataclass.
    
    Contains all configuration settings for the human-in-the-loop
    interaction system. The master 'enabled' flag controls whether
    any HITL features are active.
    
    Attributes:
        enabled: Master switch for all HITL features
        instruction_enabled: Enable instruction input from UE5
        review_enabled: Enable plan review functionality
        decision_enabled: Enable result decision functionality
        instruction_timeout: Timeout for instruction input (seconds)
        review_timeout: Timeout for plan review (seconds)
        decision_timeout: Timeout for decision requests (seconds)
        server_port: Port for the Python HTTP server
        retry_count: Number of retry attempts for failed operations
        retry_delay: Delay between retry attempts (seconds)
    """
    enabled: bool = False
    instruction_enabled: bool = True
    review_enabled: bool = True
    decision_enabled: bool = True
    instruction_timeout: float = 300.0  # 5 minutes
    review_timeout: float = 600.0       # 10 minutes
    decision_timeout: float = 120.0     # 2 minutes
    server_port: int = 8081
    retry_count: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HITLConfig':
        """
        Create an HITLConfig instance from a configuration dictionary.
        
        Args:
            data: Dictionary containing configuration values.
                  Missing keys will use default values.
                  
        Returns:
            A new HITLConfig instance with values from the dictionary.
        """
        return cls(
            enabled=data.get("enabled", False),
            instruction_enabled=data.get("instruction_enabled", True),
            review_enabled=data.get("review_enabled", True),
            decision_enabled=data.get("decision_enabled", True),
            instruction_timeout=float(data.get("instruction_timeout", 300.0)),
            review_timeout=float(data.get("review_timeout", 600.0)),
            decision_timeout=float(data.get("decision_timeout", 120.0)),
            server_port=int(data.get("server_port", 8081)),
            retry_count=int(data.get("retry_count", 3)),
            retry_delay=float(data.get("retry_delay", 1.0)),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "enabled": self.enabled,
            "instruction_enabled": self.instruction_enabled,
            "review_enabled": self.review_enabled,
            "decision_enabled": self.decision_enabled,
            "instruction_timeout": self.instruction_timeout,
            "review_timeout": self.review_timeout,
            "decision_timeout": self.decision_timeout,
            "server_port": self.server_port,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
        }


@dataclass
class InteractionState:
    """
    Interaction State dataclass.
    
    Tracks the current state of pending HITL interactions and stores
    the results of the most recent interactions.
    
    Attributes:
        instruction_pending: Whether an instruction input is pending
        review_pending: Whether a plan review is pending
        decision_pending: Whether a decision request is pending
        last_instruction: Result of the last instruction input
        last_review_result: Result of the last plan review
        last_decision: Result of the last decision request
    """
    instruction_pending: bool = False
    review_pending: bool = False
    decision_pending: bool = False
    last_instruction: Optional[Dict[str, Any]] = None
    last_review_result: Optional[Dict[str, Any]] = None
    last_decision: Optional[Dict[str, Any]] = None
    
    def clear_pending_states(self) -> None:
        """
        Clear all pending interaction states.
        
        This is typically called during error recovery to reset
        the interaction state machine.
        """
        self.instruction_pending = False
        self.review_pending = False
        self.decision_pending = False
    
    def has_any_pending(self) -> bool:
        """
        Check if any interaction is currently pending.
        
        Returns:
            True if any interaction is pending, False otherwise.
        """
        return (
            self.instruction_pending or
            self.review_pending or
            self.decision_pending
        )
