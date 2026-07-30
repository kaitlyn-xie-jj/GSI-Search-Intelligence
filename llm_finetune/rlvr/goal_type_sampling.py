#!/usr/bin/env python3
"""Helpers for deterministic goal_type-weighted RLVR sampling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping


def parse_goal_type_weights_arg(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("goal_type weights must be a JSON object")

    parsed: dict[str, float] = {}
    for key, value in data.items():
        weight = float(value)
        if weight < 0:
            raise ValueError(f"goal_type weight must be non-negative, got {key}={value}")
        parsed[str(key)] = weight
    return parsed


def normalize_goal_type_weights(
    goal_types: list[str],
    goal_type_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if not goal_types:
        return {}
    if not goal_type_weights:
        return {goal_type: 1.0 for goal_type in goal_types}

    normalized = {goal_type: max(0.0, float(goal_type_weights.get(goal_type, 0.0))) for goal_type in goal_types}
    if sum(normalized.values()) <= 0:
        return {goal_type: 1.0 for goal_type in goal_types}
    return normalized


def allocate_goal_type_counts(
    *,
    total_target: int,
    capacities: Mapping[str, int | None],
    goal_type_weights: Mapping[str, float] | None = None,
) -> dict[str, int]:
    if total_target < 0:
        raise ValueError(f"total_target must be non-negative, got {total_target}")

    result = {goal_type: 0 for goal_type in capacities}
    unlimited = any(capacity is None for capacity in capacities.values())
    remaining_target = total_target if unlimited else min(total_target, sum(int(capacity or 0) for capacity in capacities.values()))
    remaining_caps = {goal_type: capacity for goal_type, capacity in capacities.items()}

    while remaining_target > 0:
        active = [
            goal_type
            for goal_type, capacity in remaining_caps.items()
            if capacity is None or capacity > 0
        ]
        if not active:
            break

        weights = normalize_goal_type_weights(active, goal_type_weights)
        total_weight = sum(weights.values())
        raw_targets = {goal_type: remaining_target * weights[goal_type] / total_weight for goal_type in active}
        allocation = {}
        for goal_type in active:
            base = int(raw_targets[goal_type])
            capacity = remaining_caps[goal_type]
            if capacity is not None:
                base = min(base, capacity)
            allocation[goal_type] = base

        assigned = sum(allocation.values())
        leftover = remaining_target - assigned
        order = sorted(
            active,
            key=lambda goal_type: (raw_targets[goal_type] - int(raw_targets[goal_type]), weights[goal_type], goal_type),
            reverse=True,
        )
        while leftover > 0:
            progressed = False
            for goal_type in order:
                capacity = remaining_caps[goal_type]
                if capacity is not None and allocation[goal_type] >= capacity:
                    continue
                allocation[goal_type] += 1
                leftover -= 1
                progressed = True
                if leftover <= 0:
                    break
            if not progressed:
                break

        assigned = sum(allocation.values())
        if assigned <= 0:
            break
        for goal_type, count in allocation.items():
            if count <= 0:
                continue
            result[goal_type] += count
            capacity = remaining_caps[goal_type]
            if capacity is not None:
                remaining_caps[goal_type] = max(0, capacity - count)
        remaining_target -= assigned

    return result


def build_difficulty_weights_from_success_rates(
    success_rates: Mapping[str, float],
    *,
    curve: str = "power",
    gamma: float = 2.0,
    temperature: float = 2.0,
    epsilon: float = 1e-3,
    scale: float = 100.0,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for goal_type, rate in success_rates.items():
        success = max(epsilon, min(1.0 - epsilon, float(rate) / 100.0))
        difficulty = 1.0 - success
        if curve == "linear":
            weight = difficulty
        elif curve == "power":
            weight = difficulty**gamma
        elif curve == "logit":
            weight = math.log((1.0 - success) / success)
            weight = max(epsilon, weight)
        elif curve == "exp":
            weight = math.exp(temperature * difficulty)
        else:
            raise ValueError(f"unknown curve: {curve}")
        weights[str(goal_type)] = round(weight * scale, 6)
    return weights
