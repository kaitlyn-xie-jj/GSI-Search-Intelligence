#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor module - Task and event monitoring system

Provides task monitoring, event handling, and scene graph description capabilities.

Main components:
- TaskMonitor: Task monitor, manages global scene_graph and provides scene graph descriptions
- Event-driven monitoring and feedback mechanism
- Natural language scene description interface
"""

from .task_monitor import (
    TaskMonitor,
    SceneGraphDescription,
    get_global_task_monitor,
    get_scene_graph,
    start_task_monitoring,
    stop_task_monitoring,
    reset_task_monitoring
)

__all__ = [
    'TaskMonitor',
    'SceneGraphDescription',
    'get_global_task_monitor',
    'get_scene_graph',
    'start_task_monitoring',
    'stop_task_monitoring',
    'reset_task_monitoring',
]

__version__ = '1.0.0'
__author__ = 'SGI-TP Team'
__description__ = 'Task and Event Monitoring System'