"""
System Configuration Management Module.

Manages system configuration for different run modes (e.g., default, LLM finetune).
Supports loading configuration from JSON files.
"""

import json
import os
import sys
from typing import Dict, Any, List
from enum import Enum
from modules.config.base.enums import RunMode

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

class SystemConfig:
    """System configuration manager."""
    
    def __init__(self):
        self._current_mode = RunMode.DEFAULT
        self._config_cache = {}
        
        # Default configuration
        self._default_config = {
            "enable_detailed_print": True,
            "simulate_time_delay": True,
            "enable_visualization": True,
            "enable_video_recording": True,
            "fine_grained_simulation": True,
            "max_concurrency": 20,
            "enable_logging": True,
            "enable_checkpointing": True,
            "enable_new_case_generation": False,
            # Replay mode configuration block
            "replay_mode": {
                "enabled": False,       
                "trace_root": "",       
                "trace_tag": "default" 
            },
            # Human-in-the-loop configuration block
            "human_in_loop": {
                "enabled": False,
                "instruction_enabled": True,
                "review_enabled": True,
                "decision_enabled": True,
                "instruction_timeout": 300.0,
                "review_timeout": 600.0,
                "decision_timeout": 120.0,
                "server_port": 8081,
                "retry_count": 3,
                "retry_delay": 1.0
            },
        }
        
        # Config file paths for different modes
        self._config_files = {
            RunMode.DEFAULT: os.path.join(project_root, "config", "default.json"),
            RunMode.LLM_FINETUNE: os.path.join(project_root, "config", "llm_finetune.json"),
        }
        
        # Load configuration files
        self._load_configs()
    
    def _load_configs(self):
        """Load configuration from files."""
        self._mode_configs = {}
        
        # Load default configuration
        self._mode_configs[RunMode.DEFAULT] = self._default_config.copy()
        
        # Try to load default config from file
        default_config_file = self._config_files[RunMode.DEFAULT]
        if os.path.exists(default_config_file):
            try:
                with open(default_config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._mode_configs[RunMode.DEFAULT].update(file_config)
            except Exception as e:
                print(f"Warning: Unable to load default config file {default_config_file}: {e}")
        
        # Load LLM finetune mode configuration
        self._mode_configs[RunMode.LLM_FINETUNE] = self._default_config.copy()
        
        # Override with LLM finetune mode specific settings on top of defaults
        llm_finetune_overrides = {
            "enable_detailed_print": False,
            "simulate_time_delay": False,
            "enable_visualization": False,
            "enable_video_recording": False,
            "fine_grained_simulation": True,
            "max_concurrency": 50,
            "enable_logging": False,
            "enable_checkpointing": False,
            "enable_new_case_generation": False
        }
        self._mode_configs[RunMode.LLM_FINETUNE].update(llm_finetune_overrides)
        
        # Try to load LLM finetune mode config from file
        llm_finetune_config_file = self._config_files[RunMode.LLM_FINETUNE]
        if os.path.exists(llm_finetune_config_file):
            try:
                with open(llm_finetune_config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._mode_configs[RunMode.LLM_FINETUNE].update(file_config)
            except Exception as e:
                print(f"Warning: Unable to load LLM finetune config file {llm_finetune_config_file}: {e}")
    
    def set_mode(self, mode: RunMode):
        """Set the run mode."""
        self._current_mode = mode
        # Clear config cache
        self._config_cache = {}
    
    def get_mode(self) -> RunMode:
        """Get the current run mode."""
        return self._current_mode
    
    def get_config(self, key: str, default=None):
        """
        Get the value of a configuration item.
        
        Args:
            key: Configuration key name
            default: Default value
            
        Returns:
            The configuration value
        """
        # Return cached value if available
        if key in self._config_cache:
            return self._config_cache[key]
        
        # Get configuration for the current mode
        mode_config = self._mode_configs[self._current_mode]
        # Get the config value
        value = mode_config.get(key, default)
        
        # Cache the value
        self._config_cache[key] = value
        
        return value

    def set_override(self, key: str, value):
        """Override a configuration item at runtime (current mode).

        Used for dynamically modifying configuration in scenarios such as
        batch testing, e.g., enabling new case generation or setting counts.
        The override updates both the underlying config dict and the cache.

        Args:
            key: Configuration key name
            value: New value
        """
        self._mode_configs[self._current_mode][key] = value
        self._config_cache[key] = value
    
    def make_run_input_snapshot(
        self,
        *,
        mode_label: str,      
        type_name: str,
        scenario_id: str,
        goal_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a snapshot of the current run environment's input configuration
        (plain dict, can be directly passed to dump_var).
        """
        planner_mode = self.get_config("planner_mode", "full")
        robot_types = list(self.get_config("default_robot_types", ["UAV", "UGV", "Quadruped", "Humanoid"]))
        enable_new_case = bool(self.get_config("enable_new_case_generation", True))

        try:
            global_run_mode = self._current_mode.name
        except Exception:
            global_run_mode = "UNKNOWN"

        return {
            "mode": str(mode_label),
            "type_name": str(type_name),
            "scenario_id": str(scenario_id),
            "goal_id": str(goal_id),
            "planner_mode": str(planner_mode),
            "robot_type_list": robot_types,
            "enable_new_case_generation": enable_new_case,
            "global_run_mode": str(global_run_mode),
        }
    
    def __getattr__(self, name: str):
        """Support config.xxx style access to configuration items."""
        config_key = name
        if name.startswith('is_') or name.startswith('should_'):
            # Keep is_xxx and should_xxx naming as-is
            pass
        elif name.startswith('get_'):
            # Strip the get_ prefix
            config_key = name[4:]
        
        # Get the config value
        return self.get_config(config_key)
    
    def reload_configs(self):
        """Reload configuration files."""
        self._load_configs()
        # Clear config cache
        self._config_cache = {}
    
    def get_human_in_loop_config(self) -> Dict[str, Any]:
        """
        Get the human-in-the-loop configuration.
        
        Returns:
            HITL configuration dictionary containing enabled, instruction_enabled, etc.
        """
        return self.get_config("human_in_loop", {
            "enabled": False,
            "instruction_enabled": True,
            "review_enabled": True,
            "decision_enabled": True,
            "instruction_timeout": 300.0,
            "review_timeout": 600.0,
            "decision_timeout": 120.0,
            "server_port": 8081,
            "retry_count": 3,
            "retry_delay": 1.0
        })

# Create global config instance
config = SystemConfig()

def set_run_mode(mode: RunMode):
    """Set the run mode."""
    config.set_mode(mode)

def get_run_mode() -> RunMode:
    """Get the run mode."""
    return config.get_mode()
