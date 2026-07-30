#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template utility module.

Provides template update, validation, and management features.
All template update operations go through strict validation and return structured results.
"""

import copy
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class ValidationLevel(Enum):
    """Validation level enum."""
    BASIC = "basic"          # Basic validation: check required fields.
    STANDARD = "standard"    # Standard validation: basic validation plus type checks.
    STRICT = "strict"        # Strict validation: standard validation plus business rule checks.


class OperationType(Enum):
    """Operation type enum."""
    CREATE = "create"        # Create template.
    UPDATE = "update"        # Update template.
    DELETE = "delete"        # Delete template.
    VALIDATE = "validate"    # Validate template.


@dataclass
class ValidationResult:
    """Validation result data class."""
    is_valid: bool                          # Whether validation passed.
    errors: List[str]                       # Error messages.
    warnings: List[str]                     # Warning messages.
    validation_level: ValidationLevel       # Validation level.
    timestamp: datetime                     # Validation timestamp.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "validation_level": self.validation_level.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OperationResult:
    """Operation result data class."""
    success: bool                           # Whether the operation succeeded.
    operation_type: OperationType           # Operation type.
    template_category: str                  # Template category.
    template_type: str                      # Template type.
    validation_result: ValidationResult     # Validation result.
    data: Optional[Dict[str, Any]] = None   # Operation result data.
    message: str = ""                       # Operation message.
    timestamp: datetime = None              # Operation timestamp.
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "operation_type": self.operation_type.value,
            "template_category": self.template_category,
            "template_type": self.template_type,
            "validation_result": self.validation_result.to_dict(),
            "data": self.data,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


class TemplateValidator:
    """Template validator class."""
    
    # Required field definitions.
    REQUIRED_FIELDS = {
        "robot": ["category", "type", "description"],
        "prop": ["category", "type", "description"],
        "building": ["category", "type", "description", "size"],
        "situation": ["category", "type", "description"]
    }
    
    # Field type definitions.
    FIELD_TYPES = {
        "category": str,
        "type": str,
        "description": str,
        "size": (tuple, list),
        "attributes": dict,
        "status": dict,
        "skills": list
    }
    
    @classmethod
    def validate_template(
        cls, 
        template: Dict[str, Any], 
        category: str,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationResult:
        """
        Validate a template.
        
        Args:
            template: Template to validate.
            category: Template category.
            validation_level: Validation level.
            
        Returns:
            ValidationResult: Validation result.
        """
        errors = []
        warnings = []
        timestamp = datetime.now()
        
        try:
            # Basic validation.
            if validation_level.value in ["basic", "standard", "strict"]:
                errors.extend(cls._validate_basic(template, category))
            
            # Standard validation.
            if validation_level.value in ["standard", "strict"]:
                errors.extend(cls._validate_types(template))
                warnings.extend(cls._validate_optional_fields(template, category))
            
            # Strict validation.
            if validation_level.value == "strict":
                errors.extend(cls._validate_business_rules(template, category))
            
        except Exception as e:
            errors.append(f"Exception occurred during validation: {str(e)}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validation_level=validation_level,
            timestamp=timestamp
        )
    
    @classmethod
    def _validate_basic(cls, template: Dict[str, Any], category: str) -> List[str]:
        """Basic validation: check required fields."""
        errors = []
        
        if not isinstance(template, dict):
            errors.append("Template must be a dict type")
            return errors
        
        required_fields = cls.REQUIRED_FIELDS.get(category, [])
        for field in required_fields:
            if field not in template:
                errors.append(f"Missing required field: {field}")
            elif not template[field]:
                errors.append(f"Required field cannot be empty: {field}")
        
        return errors
    
    @classmethod
    def _validate_types(cls, template: Dict[str, Any]) -> List[str]:
        """Type validation: check field types."""
        errors = []
        
        for field, value in template.items():
            if field in cls.FIELD_TYPES:
                expected_type = cls.FIELD_TYPES[field]
                if not isinstance(value, expected_type):
                    # Handle display for tuple types.
                    if isinstance(expected_type, tuple):
                        type_names = [t.__name__ for t in expected_type]
                        expected_name = " or ".join(type_names)
                    else:
                        expected_name = expected_type.__name__
                    
                    errors.append(f"Field {field} type error, expected {expected_name}, got {type(value).__name__}")
        
        return errors
    
    @classmethod
    def _validate_optional_fields(cls, template: Dict[str, Any], category: str) -> List[str]:
        """Optional field validation: check optional field reasonableness."""
        warnings = []
        
        # Check building size.
        if category == "building" and "size" in template:
            size = template["size"]
            if isinstance(size, (tuple, list)) and len(size) == 2:
                width, height = size
                if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
                    warnings.append("Building size should be numeric type")
                elif width <= 0 or height <= 0:
                    warnings.append("Building size should be positive")
        
        # Check skill list.
        if "skills" in template:
            skills = template["skills"]
            if isinstance(skills, list):
                for skill in skills:
                    if not isinstance(skill, str):
                        warnings.append(f"Skill name should be string type: {skill}")
        
        return warnings
    
    @classmethod
    def _validate_business_rules(cls, template: Dict[str, Any], category: str) -> List[str]:
        """Business rule validation: check business logic."""
        errors = []
        
        # Validate category and type consistency.
        if "category" in template and "type" in template:
            if template["category"] != category:
                errors.append(f"Template category mismatch: expected {category}, got {template['category']}")
        
        # Validate building-specific rules.
        if category == "building":
            if "size" in template:
                size = template["size"]
                if isinstance(size, (tuple, list)) and len(size) == 2:
                    width, height = size
                    if width > 1000 or height > 1000:
                        errors.append("Building size too large (exceeds 1000)")
        
        # Validate robot-specific rules.
        if category == "robot":
            if "skills" in template:
                skills = template["skills"]
                if isinstance(skills, list) and len(skills) == 0:
                    errors.append("Robot must have at least one skill")
        
        return errors


class TemplateManager:
    """Template manager class."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize the template manager.
        
        Args:
            validation_level: Default validation level.
        """
        self.validation_level = validation_level
        self.validator = TemplateValidator()
        self._operation_history: List[OperationResult] = []
    
    def create_template(
        self, 
        category: str, 
        template_type: str, 
        template_data: Dict[str, Any],
        validation_level: Optional[ValidationLevel] = None
    ) -> OperationResult:
        """
        Create a new template.
        
        Args:
            category: Template category.
            template_type: Template type.
            template_data: Template data.
            validation_level: Validation level, optional.
            
        Returns:
            OperationResult: Operation result.
        """
        level = validation_level or self.validation_level
        
        # Validate template.
        validation_result = self.validator.validate_template(
            template_data, category, level
        )
        
        # Create operation result.
        if validation_result.is_valid:
            # Deep-copy template data to avoid accidental changes.
            safe_template = copy.deepcopy(template_data)
            
            result = OperationResult(
                success=True,
                operation_type=OperationType.CREATE,
                template_category=category,
                template_type=template_type,
                validation_result=validation_result,
                data=safe_template,
                message=f"Successfully created template {category}:{template_type}"
            )
        else:
            result = OperationResult(
                success=False,
                operation_type=OperationType.CREATE,
                template_category=category,
                template_type=template_type,
                validation_result=validation_result,
                message=f"Failed to create template: {', '.join(validation_result.errors)}"
            )
        
        # Record operation history.
        self._operation_history.append(result)
        
        return result
    
    def update_template(
        self, 
        category: str, 
        template_type: str, 
        updates: Dict[str, Any],
        current_template: Optional[Dict[str, Any]] = None,
        validation_level: Optional[ValidationLevel] = None
    ) -> OperationResult:
        """
        Update a template.
        
        Args:
            category: Template category.
            template_type: Template type.
            updates: Update data.
            current_template: Current template data, optional.
            validation_level: Validation level, optional.
            
        Returns:
            OperationResult: Operation result.
        """
        level = validation_level or self.validation_level
        
        # If current_template is provided, merge updates into it.
        if current_template:
            updated_template = copy.deepcopy(current_template)
            updated_template.update(updates)
        else:
            updated_template = copy.deepcopy(updates)
        
        # Validate the updated template.
        validation_result = self.validator.validate_template(
            updated_template, category, level
        )
        
        # Create operation result.
        if validation_result.is_valid:
            result = OperationResult(
                success=True,
                operation_type=OperationType.UPDATE,
                template_category=category,
                template_type=template_type,
                validation_result=validation_result,
                data=updated_template,
                message=f"Successfully updated template {category}:{template_type}"
            )
        else:
            result = OperationResult(
                success=False,
                operation_type=OperationType.UPDATE,
                template_category=category,
                template_type=template_type,
                validation_result=validation_result,
                message=f"Failed to update template: {', '.join(validation_result.errors)}"
            )
        
        # Record operation history.
        self._operation_history.append(result)
        
        return result
    
    def validate_template_only(
        self, 
        template: Dict[str, Any], 
        category: str,
        validation_level: Optional[ValidationLevel] = None
    ) -> OperationResult:
        """
        Validate a template only, without performing other operations.
        
        Args:
            template: Template to validate.
            category: Template category.
            validation_level: Validation level, optional.
            
        Returns:
            OperationResult: Operation result.
        """
        level = validation_level or self.validation_level
        
        # Validate template.
        validation_result = self.validator.validate_template(
            template, category, level
        )
        
        # Create operation result.
        template_type = template.get("type", "unknown")
        result = OperationResult(
            success=validation_result.is_valid,
            operation_type=OperationType.VALIDATE,
            template_category=category,
            template_type=template_type,
            validation_result=validation_result,
            data=template if validation_result.is_valid else None,
            message="Validation passed" if validation_result.is_valid else f"Validation failed: {', '.join(validation_result.errors)}"
        )
        
        # Record operation history.
        self._operation_history.append(result)
        
        return result
    
    def get_operation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get operation history.
        
        Args:
            limit: Maximum number of records to return, optional.
            
        Returns:
            List[Dict[str, Any]]: Operation history list.
        """
        history = self._operation_history
        if limit:
            history = history[-limit:]
        
        return [op.to_dict() for op in history]
    
    def clear_operation_history(self) -> None:
        """Clear operation history."""
        self._operation_history.clear()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get validation summary statistics.
        
        Returns:
            Dict[str, Any]: Validation summary.
        """
        total_operations = len(self._operation_history)
        successful_operations = sum(1 for op in self._operation_history if op.success)
        failed_operations = total_operations - successful_operations
        
        # Statistics by operation type.
        operation_stats = {}
        for op in self._operation_history:
            op_type = op.operation_type.value
            if op_type not in operation_stats:
                operation_stats[op_type] = {"total": 0, "success": 0, "failed": 0}
            
            operation_stats[op_type]["total"] += 1
            if op.success:
                operation_stats[op_type]["success"] += 1
            else:
                operation_stats[op_type]["failed"] += 1
        
        # Statistics by category.
        category_stats = {}
        for op in self._operation_history:
            category = op.template_category
            if category not in category_stats:
                category_stats[category] = {"total": 0, "success": 0, "failed": 0}
            
            category_stats[category]["total"] += 1
            if op.success:
                category_stats[category]["success"] += 1
            else:
                category_stats[category]["failed"] += 1
        
        return {
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
            "operation_stats": operation_stats,
            "category_stats": category_stats,
            "default_validation_level": self.validation_level.value
        }


# Create the global template manager instance.
template_manager = TemplateManager()


# Convenience functions.
def create_template(
    category: str, 
    template_type: str, 
    template_data: Dict[str, Any],
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> OperationResult:
    """
    Convenience function for creating a template.
    
    Args:
        category: Template category.
        template_type: Template type.
        template_data: Template data.
        validation_level: Validation level.
        
    Returns:
        OperationResult: Operation result.
    """
    return template_manager.create_template(category, template_type, template_data, validation_level)


def update_template(
    category: str, 
    template_type: str, 
    updates: Dict[str, Any],
    current_template: Optional[Dict[str, Any]] = None,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> OperationResult:
    """
    Convenience function for updating a template.
    
    Args:
        category: Template category.
        template_type: Template type.
        updates: Update data.
        current_template: Current template data, optional.
        validation_level: Validation level.
        
    Returns:
        OperationResult: Operation result.
    """
    return template_manager.update_template(category, template_type, updates, current_template, validation_level)


def validate_template(
    template: Dict[str, Any], 
    category: str,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> OperationResult:
    """
    Convenience function for validating a template.
    
    Args:
        template: Template to validate.
        category: Template category.
        validation_level: Validation level.
        
    Returns:
        OperationResult: Operation result.
    """
    return template_manager.validate_template_only(template, category, validation_level)
