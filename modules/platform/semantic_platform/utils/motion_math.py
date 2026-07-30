# motion_math.py
# General utility functions for numerical/interpolation/time operations

from typing import List
import numpy as np


def clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smooth_step(t: float) -> float:
    """
    Smooth step (S-curve), input doesn't need pre-clamping, function clamps internally.
    Commonly used for smooth interpolation of pose/velocity in 0..1 range.
    """
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def smooth_step_derivative(t: float) -> float:
    """
    First derivative of smooth step, used to estimate velocity/climb rate.
    """
    t = clamp01(t)
    return 6.0 * t * (1.0 - t)


def interpolate_position(start: List[float], end: List[float], t: float) -> List[float]:
    """
    Linear interpolation for 2D position.
    """
    t = float(t)
    return [
        float(start[0] + (end[0] - start[0]) * t),
        float(start[1] + (end[1] - start[1]) * t),
    ]


def heading(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate heading angle (radians) from two points a->b.
    """
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if np.linalg.norm(d) < 1e-9:
        return 0.0
    return float(np.arctan2(d[1], d[0]))


def calculate_path_length(path: List[np.ndarray]) -> float:
    """
    Calculate total path length (sum of polyline segments).
    """
    if not path or len(path) == 1:
        return 0.0
    length = 0.0
    for i in range(1, len(path)):
        length += float(np.linalg.norm(np.asarray(path[i], dtype=float) - np.asarray(path[i - 1], dtype=float)))
    return length
