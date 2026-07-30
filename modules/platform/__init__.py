# -*- coding: utf-8 -*-
"""
Platform Abstraction Layer - 平台抽象层

提供多平台支持的统一接口，使三层任务规划架构能够同时支持
语义平台（Semantic Platform）和虚幻平台（Unreal Platform）。

核心组件：
- AbstractSceneGraph: 场景图抽象基类
- AbstractPlatformExecutor: 平台执行器抽象基类
- PlatformType: 平台类型枚举
- Platform Factory: 平台工厂函数

使用示例：
    from modules.platform import (
        initialize_platform,
        get_scene_graph,
        get_platform_executor,
        cleanup_platform,
        PlatformType,
    )
    
    # 初始化语义平台
    await initialize_platform(
        PlatformType.SEMANTIC,
        initial_nodes=nodes,
        initial_edges=edges,
        initial_goal=goal,
    )
    
    # 获取场景图
    scene_graph = get_scene_graph()
    
    # 清理平台
    await cleanup_platform()
"""

# 抽象基类
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.abstract_platform_executor import AbstractPlatformExecutor

# 平台工厂
from modules.platform.platform_factory import (
    PlatformType,
    initialize_platform,
    get_scene_graph,
    get_platform_executor,
    get_platform_type,
    create_platform_executor,
    cleanup_platform,
    reset_platform,
)

# 数据类型
from modules.platform.data_types import (
    ShapePoint,
    ShapeCircle,
    ShapeRectangle,
    ShapePolygon,
    ShapeLinestring,
    ShapeData,
    NodeProperties,
    NodeData,
    EdgeData,
    SceneGraphData,
    GoalData,
    NodeCategory,
    RobotType,
    ShapeType,
    TransFacilityType,
)

__all__ = [
    # 抽象基类
    "AbstractSceneGraph",
    "AbstractPlatformExecutor",
    # 平台工厂
    "PlatformType",
    "initialize_platform",
    "get_scene_graph",
    "get_platform_executor",
    "get_platform_type",
    "create_platform_executor",
    "cleanup_platform",
    "reset_platform",
    # 数据类型
    "ShapePoint",
    "ShapeCircle",
    "ShapeRectangle",
    "ShapePolygon",
    "ShapeLinestring",
    "ShapeData",
    "NodeProperties",
    "NodeData",
    "EdgeData",
    "SceneGraphData",
    "GoalData",
    "NodeCategory",
    "RobotType",
    "ShapeType",
    "TransFacilityType",
]
