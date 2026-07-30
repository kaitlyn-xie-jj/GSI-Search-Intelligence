# -*- coding: utf-8 -*-
"""
TaskAllocator - LP-based robot-skill allocation module

Computes allocation weights based on distance between robots and skill target positions,
iteratively assigns skills from the DAG.
Supports heterogeneous multi-robot: each typed skill (robot_type:skill_str) can only be assigned to a type-matching robot.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, lpSum

logger = logging.getLogger(__name__)


def extract_robot_type(typed_skill: str) -> Optional[str]:
    """Extract robot_type from a typed skill string.

    Args:
        typed_skill: e.g. "UAV:take_off" or "take_off".

    Returns:
        robot_type string, or None if no prefix.
    """
    if ":" in typed_skill:
        return typed_skill.partition(":")[0].strip()
    return None


def extract_skill_str(typed_skill: str) -> str:
    """Extract the pure skill_str from a typed skill string.

    Args:
        typed_skill: e.g. "UAV:take_off" or "take_off".

    Returns:
        Pure skill string without robot_type prefix.
    """
    if ":" in typed_skill:
        return typed_skill.partition(":")[2].strip()
    return typed_skill.strip()


class TaskAllocator:
    """Task allocator — LP-based robot-skill assignment.

    Core flow:
    1. Find root skills (no in-edges) from the DAG
    2. Compute distance-based weights from each robot to each root skill target
    3. Solve LP assignment using PuLP
    4. Remove assigned nodes, repeat until all skills are assigned
    5. Each allocation round corresponds to one timestep

    Attributes:
        alpha: Distance weight factor controlling how much distance affects allocation.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    # =========================================================================
    # Public API
    # =========================================================================

    def allocate(
        self,
        graph: nx.DiGraph,
        robot_labels: List[str],
        robot_positions: Dict[str, np.ndarray],
        skill_positions: Dict[str, np.ndarray],
        robot_type_map: Optional[Dict[str, str]] = None,
    ) -> Dict[int, Dict[str, str]]:
        """Iteratively assign all skills to robots, generating multiple timesteps by DAG layers.

        Supports heterogeneous multi-robot: typed skills (robot_type:skill_str) can only be
        assigned to type-matching robots.

        Algorithm:
        1. Find current DAG root skills (no in-edges)
        2. Use LP to assign root skills to robots (type constraint + one skill per robot + one robot per skill)
        3. Remove assigned skills from DAG
        4. Repeat until all skills are assigned

        Args:
            graph: Skill dependency DAG (networkx DiGraph), nodes are typed skill strings.
            robot_labels: List of available robot labels.
            robot_positions: {robot_label: np.ndarray([x, y])} robot positions.
            skill_positions: {typed_skill: np.ndarray([x, y])} skill target positions.
            robot_type_map: {robot_label: robot_type} mapping for type constraints.
                If None, no type constraints are applied (backward compatible).

        Returns:
            {timestep: {robot_label: typed_skill}} allocation result by timestep.
        """
        if graph.number_of_nodes() == 0:
            return {}

        remaining = graph.copy()
        timestep = 0
        result: Dict[int, Dict[str, str]] = {}

        while remaining.number_of_nodes() > 0:
            roots = self.get_root_skills(remaining)
            if not roots:
                logger.error("No root skills found but graph is non-empty; breaking.")
                break

            # Compute weights and solve LP (with type constraints)
            weights = self._calculate_weights(
                robot_positions, skill_positions, roots, robot_labels
            )
            eligible = self._build_eligibility(roots, robot_labels, robot_type_map)
            assignments = self._solve_lp(roots, robot_labels, weights, eligible)

            # Record this round's assignments
            ts_alloc: Dict[str, str] = {}
            assigned_skills = set()
            for robot, skill in assignments:
                ts_alloc[robot] = skill
                assigned_skills.add(skill)

            if ts_alloc:
                result[timestep] = ts_alloc
                timestep += 1

            # Remove assigned skill nodes
            if not assigned_skills:
                logger.warning(
                    "LP solver assigned no skills this round. "
                    "Falling back to greedy assignment for remaining roots."
                )
                assigned_skills = self._greedy_fallback(
                    roots, robot_labels, robot_type_map, result, timestep
                )
                if assigned_skills:
                    timestep += 1
                else:
                    break

            remaining.remove_nodes_from(list(assigned_skills))

        return result

    # =========================================================================
    # DAG utilities
    # =========================================================================

    @staticmethod
    def get_root_skills(graph: nx.DiGraph) -> List[str]:
        """Get root skills with no in-edges.

        Args:
            graph: Skill dependency DAG.

        Returns:
            List of nodes with no in-edges.
        """
        return [n for n in graph.nodes if graph.in_degree(n) == 0]

    # =========================================================================
    # Type eligibility
    # =========================================================================

    @staticmethod
    def _build_eligibility(
        skills: List[str],
        robots: List[str],
        robot_type_map: Optional[Dict[str, str]],
    ) -> Dict[Tuple[str, str], bool]:
        """Build robot-skill type compatibility matrix.

        Args:
            skills: List of typed skills (robot_type:skill_str format).
            robots: List of robot labels.
            robot_type_map: {label: type} mapping. If None, all pairs are compatible.

        Returns:
            {(robot, skill): bool} compatibility dictionary.
        """
        eligible: Dict[Tuple[str, str], bool] = {}
        for r in robots:
            for s in skills:
                if robot_type_map is None:
                    eligible[(r, s)] = True
                else:
                    required_type = extract_robot_type(s)
                    robot_type = robot_type_map.get(r)
                    # If skill has no type prefix, any robot can execute it
                    eligible[(r, s)] = (
                        required_type is None or required_type == robot_type
                    )
        return eligible

    def _greedy_fallback(
        self,
        roots: List[str],
        robot_labels: List[str],
        robot_type_map: Optional[Dict[str, str]],
        result: Dict[int, Dict[str, str]],
        timestep: int,
    ) -> set:
        """Greedy fallback assignment (used when LP fails), respects type constraints.

        Args:
            roots: Root skills to assign this round.
            robot_labels: List of available robot labels.
            robot_type_map: {label: type} mapping.
            result: Current allocation result dict (modified in-place).
            timestep: Current timestep.

        Returns:
            Set of assigned skills.
        """
        assigned_skills = set()
        used_robots = set()
        for skill in roots:
            required_type = extract_robot_type(skill)
            for robot in robot_labels:
                if robot in used_robots:
                    continue
                if robot_type_map and required_type:
                    if robot_type_map.get(robot) != required_type:
                        continue
                result.setdefault(timestep, {})[robot] = skill
                assigned_skills.add(skill)
                used_robots.add(robot)
                break
        return assigned_skills

    # =========================================================================
    # Weight calculation
    # =========================================================================

    def _calculate_weights(
        self,
        robot_positions: Dict[str, np.ndarray],
        skill_positions: Dict[str, np.ndarray],
        skills: List[str],
        robots: List[str],
    ) -> Dict[Tuple[str, str], float]:
        """Compute robot-skill allocation weights.

        Formula: w(robot, skill) = 1 - alpha * d'(robot, skill)
        where d' = distance / max_distance (normalized distance)

        Args:
            robot_positions: {robot_label: np.ndarray([x, y])}.
            skill_positions: {skill_str: np.ndarray([x, y])}.
            skills: Skills to assign this round.
            robots: Available robot labels.

        Returns:
            {(robot, skill): weight} weight dictionary.
        """
        # Compute all distances
        distances: Dict[Tuple[str, str], float] = {}
        for r in robots:
            r_pos = robot_positions.get(r, np.array([0.0, 0.0]))
            for s in skills:
                s_pos = skill_positions.get(s, np.array([0.0, 0.0]))
                distances[(r, s)] = float(np.linalg.norm(r_pos - s_pos))

        # Normalize
        max_dist = max(distances.values()) if distances else 1.0
        if max_dist == 0:
            max_dist = 1.0  # Avoid division by zero

        weights: Dict[Tuple[str, str], float] = {}
        for (r, s), d in distances.items():
            d_norm = d / max_dist
            weights[(r, s)] = 1.0 - self.alpha * d_norm

        return weights

    # =========================================================================
    # LP solver
    # =========================================================================

    @staticmethod
    def _solve_lp(
        skills: List[str],
        robots: List[str],
        weights: Dict[Tuple[str, str], float],
        eligible: Optional[Dict[Tuple[str, str], bool]] = None,
    ) -> List[Tuple[str, str]]:
        """Solve LP assignment problem (with type constraints).

        Objective: maximize sum(w(r,s) * x(r,s))
        Constraints:
        - Each skill assigned to at most one robot
        - Each robot assigned at most one skill
        - Only type-compatible (robot, skill) pairs can be assigned

        Args:
            skills: Skills to assign.
            robots: Available robots.
            weights: {(robot, skill): weight} weight dictionary.
            eligible: {(robot, skill): bool} type compatibility matrix.
                If None, no type constraints are applied.

        Returns:
            [(robot, skill), ...] assignment result list.
        """
        # Filter to compatible (robot, skill) pairs
        pairs = [
            (r, s) for r in robots for s in skills
            if eligible is None or eligible.get((r, s), True)
        ]
        if not pairs:
            return []

        prob = LpProblem("SkillAssignment", LpMaximize)

        # Decision variables (only for compatible pairs)
        x = {
            (r, s): LpVariable(f"x_{r}_{s}", cat=LpBinary)
            for r, s in pairs
        }

        # Objective: maximize weighted assignment
        prob += lpSum(weights.get((r, s), 0.0) * x[(r, s)] for r, s in pairs)

        # Constraint 1: each skill assigned to at most one robot
        for s in skills:
            eligible_vars = [x[(r, s)] for r in robots if (r, s) in x]
            if eligible_vars:
                prob += lpSum(eligible_vars) <= 1

        # Constraint 2: each robot assigned at most one skill
        for r in robots:
            eligible_vars = [x[(r, s)] for s in skills if (r, s) in x]
            if eligible_vars:
                prob += lpSum(eligible_vars) <= 1

        # Solve (silent mode)
        from pulp import PULP_CBC_CMD
        prob.solve(PULP_CBC_CMD(msg=0))

        # Extract assignment results
        assignments = [
            (r, s)
            for r, s in pairs
            if x[(r, s)].value() == 1
        ]
        return assignments

    # =========================================================================
    # Position extraction
    # =========================================================================

    @staticmethod
    def _extract_skill_position(
        typed_skill: str,
        real_time_pos_map: Optional[Dict],
    ) -> np.ndarray:
        """Extract target position from a typed skill string and world model.

        Parses the location parameter (first angle-bracket argument) from the skill string,
        then looks up the corresponding position in real_time_pos_map.
        Supports both typed format (robot_type:skill_str) and plain skill_str format.

        Args:
            typed_skill: Typed skill string, e.g. "UAV:navigate<Hotel-1>",
                         "UGV:search<cybertown>_for<target>", or "navigate<Hotel-1>".
            real_time_pos_map: Position mapping with structure
                {category: {subcategory: {label: [x, y]}}}.

        Returns:
            np.ndarray([x, y]) target position; defaults to [0.0, 0.0] if not found.
        """
        default_pos = np.array([0.0, 0.0])

        # Extract pure skill_str
        skill_str = extract_skill_str(typed_skill)

        # Extract first angle-bracket argument as location label
        match = re.search(r"<([^>]+)>", skill_str)
        if not match:
            return default_pos

        location_label = match.group(1).strip()

        if not real_time_pos_map:
            return default_pos

        # Look up in pos_map (iterate all categories and subcategories)
        for category_data in real_time_pos_map.values():
            if not isinstance(category_data, dict):
                continue
            for subcategory_data in category_data.values():
                if not isinstance(subcategory_data, dict):
                    continue
                if location_label in subcategory_data:
                    pos = subcategory_data[location_label]
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        return np.array([float(pos[0]), float(pos[1])])

        return default_pos

    @classmethod
    def build_skill_positions(
        cls,
        skill_list: List[str],
        real_time_pos_map: Optional[Dict],
    ) -> Dict[str, np.ndarray]:
        """Extract target positions for each skill in the list.

        Args:
            skill_list: List of skill strings.
            real_time_pos_map: Position mapping.

        Returns:
            {skill_str: np.ndarray([x, y])} skill position dictionary.
        """
        return {
            skill: cls._extract_skill_position(skill, real_time_pos_map)
            for skill in skill_list
        }

    @staticmethod
    def build_robot_positions(
        robot_labels: List[str],
        real_time_pos_map: Optional[Dict],
    ) -> Dict[str, np.ndarray]:
        """Extract robot positions from real_time_pos_map.

        Args:
            robot_labels: List of robot labels.
            real_time_pos_map: Position mapping.

        Returns:
            {robot_label: np.ndarray([x, y])} robot position dictionary.
        """
        positions: Dict[str, np.ndarray] = {}
        default_pos = np.array([0.0, 0.0])

        if not real_time_pos_map:
            return {r: default_pos.copy() for r in robot_labels}

        # Robots are typically under the "robot" category
        for robot_label in robot_labels:
            found = False
            for category_data in real_time_pos_map.values():
                if not isinstance(category_data, dict):
                    continue
                for subcategory_data in category_data.values():
                    if not isinstance(subcategory_data, dict):
                        continue
                    if robot_label in subcategory_data:
                        pos = subcategory_data[robot_label]
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            positions[robot_label] = np.array(
                                [float(pos[0]), float(pos[1])]
                            )
                            found = True
                            break
                if found:
                    break
            if not found:
                positions[robot_label] = default_pos.copy()

        return positions
