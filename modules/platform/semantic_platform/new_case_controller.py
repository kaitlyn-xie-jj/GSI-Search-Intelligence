# -*- coding: utf-8 -*-
"""
General new-case injection controller - geometric distribution + dynamic budget + cooldown + fallback.

Design goals:
  - Independent of the planning method; depends only on skills_by_timestep from the world model.
  - Keep injection timing statistically well distributed.
  - Preserve skill-balanced and instance-balanced selection.
  - Use a cooldown window to prevent repeated events from consecutive injections.
  - Deduplicate plan fingerprints to avoid repeated injections on similar replanned plans.
  - Use a fallback mechanism to spend the quota when possible.

Core algorithm:
  When each plan round arrives, compute injection probability p = n_remaining / window,
  where window = n_remaining * spacing_factor.
  spacing_factor controls injection density (default 2 = one injection every 2 rounds on average).
  During cooldown, force p to 0. If plan fingerprints overlap heavily, dampen p.
  Force fallback injection after multiple consecutive rounds without injection.
"""
from typing import Dict, List, Tuple, Optional
import random
from modules.config.entities.new_case_config import NEW_CASE_OP_TEMPLATES, NEW_CASE_EVENT_TEMPLATES
from modules.config.base.enums import SkillName
from modules.utils.system.var_dump import dump_var


class NewCaseController:
    """
    New-case controller for one experiment (one solve_task cycle).

    Uses a "geometric distribution + dynamic budget + cooldown + fallback" strategy:
    - Quota (n_new_max)
    - Online geometric-distribution injection: p = n_remaining / window
    - Cooldown window: skip cooldown_rounds rounds after injection
    - Plan fingerprint deduplication: lower injection probability for similar plans
    - Fallback mechanism: force injection after multiple consecutive rounds without injection
    - Skill-balanced and instance-balanced selection
    - Stats: total, by skill, by new-case type, and by round
    """

    def __init__(
        self,
        n_new_max: int = 5,
        enable_generation: bool = True,
        is_replay: bool = False,
        rng: Optional[random.Random] = None,
        spacing_factor: float = 2.0,
        cooldown_rounds: int = 2,
        similarity_threshold: float = 0.8,
        similarity_damping: float = 0.3,
    ):
        self.n_new_max = max(0, int(n_new_max))
        self.enable_generation = bool(enable_generation)
        self.is_replay = is_replay
        self._replay_meta = None

        self._rng = rng or random

        # ---- Geometric distribution parameters ----
        self.spacing_factor = max(1.0, float(spacing_factor))
        self.cooldown_rounds = max(0, int(cooldown_rounds))
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_damping = float(similarity_damping)

        # ---- State ----
        self.generated_count = 0           # Total generated in this experiment
        self._round_index = 0              # Current plan round
        self._cooldown_remaining = 0       # Remaining cooldown rounds
        self._rounds_since_last_injection = 0  # Rounds since last injection
        self._last_plan_fingerprint: Optional[frozenset] = None  # Skill fingerprint from the previous plan

        # ---- Skill counts (stable order) ----
        self._skill_order: List[str] = [s.value for s in SkillName]
        self._skill_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self._skill_order)}
        self._skill_counts: List[int] = [0] * len(self._skill_order)

        # ---- New-case type counts (stable order, by op.code) ----
        self._op_order: List[str] = [op["code"] for op in NEW_CASE_OP_TEMPLATES]
        self._op_to_idx: Dict[str, int] = {code: i for i, code in enumerate(self._op_order)}
        self._op_counts: List[int] = [0] * len(self._op_order)
        self._no_op_bucket = "event_without_op"
        self._no_op_count = 0

        # ---- Unique instance selected for the current plan ----
        self._current_plan_selection: Optional[Tuple[int, str, str]] = None

        # ---- Blacklisted skills ----
        self._blacklist_skills = {"sync_wait"}

        # ---- Round-level injection log (for summary) ----
        self._round_injection_log: List[int] = []  # Records the round where each injection happened

    # =========================================================================
    # Plan-Level Lifecycle
    # =========================================================================

    def reset_for_new_plan(self):
        """Call when each new plan round starts."""
        self._current_plan_selection = None

    def set_replay_meta(self, meta):
        self._replay_meta = meta

    # =========================================================================
    # Core Decision: Whether to Inject This Round
    # =========================================================================

    def should_attempt_injection_this_plan(self) -> bool:
        """Check the global quota and enable switch."""
        return self.enable_generation and (self.generated_count < self.n_new_max)

    def _compute_plan_fingerprint(self, plan: Dict) -> frozenset:
        """
        Extract the skill fingerprint from skills_by_timestep:
        frozenset of (robot_label, skill_name)
        """
        pairs = set()
        for _t_str, robots in (plan or {}).items():
            for rlabel, spec in (robots or {}).items():
                skill = (spec or {}).get("skill")
                if skill and skill not in self._blacklist_skills:
                    pairs.add((rlabel, skill))
        return frozenset(pairs)

    def _plan_similarity(self, fp_a: Optional[frozenset], fp_b: Optional[frozenset]) -> float:
        """Jaccard similarity."""
        if not fp_a or not fp_b:
            return 0.0
        union = fp_a | fp_b
        if not union:
            return 0.0
        return len(fp_a & fp_b) / len(union)

    def _should_inject_this_round(self, plan: Dict) -> bool:
        """
        Core decision for geometric distribution + cooldown + fingerprint deduplication + fallback.

        Called inside select_skill_instance, once per plan round.
        """
        n_remaining = self.n_new_max - self.generated_count
        if n_remaining <= 0:
            return False

        # ---- Dynamic window ----
        window = n_remaining * self.spacing_factor
        p = n_remaining / max(window, n_remaining)  # p in (0, 1]

        # ---- Cooldown ----
        if self._cooldown_remaining > 0:
            p = 0.0

        # ---- Plan fingerprint deduplication ----
        current_fp = self._compute_plan_fingerprint(plan)
        if p > 0.0 and self._last_plan_fingerprint is not None:
            sim = self._plan_similarity(current_fp, self._last_plan_fingerprint)
            if sim > self.similarity_threshold:
                p *= self.similarity_damping

        # ---- Fallback: force injection after 2 consecutive rounds without injection ----
        force_threshold = 2
        if self._rounds_since_last_injection >= force_threshold and n_remaining > 0:
            p = 1.0

        # ---- Roll the dice ----
        inject = self._rng.random() < p

        # Update the fingerprint cache either way for the next-round comparison
        self._last_plan_fingerprint = current_fp

        return inject

    # =========================================================================
    # Instance Selection
    # =========================================================================

    def select_skill_instance(self, plan: Dict) -> Optional[Tuple[int, str, str]]:
        """
        Select one instance from the plan: (timestep, robot_label, skill_name).

        Combines injection decision, skill-balanced selection, and instance-balanced selection.
        Called once by skill_executor.execute_plan() when each new plan arrives.
        """
        # ---- Update round counters ----
        self._round_index += 1
        self._rounds_since_last_injection += 1
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # ---- Replay mode ----
        if self.is_replay and self._replay_meta and self._replay_meta.get("current_plan_selection") is not None:
            self._current_plan_selection = tuple(self._replay_meta["current_plan_selection"])
            dump_var("current_plan_selection", self._current_plan_selection)
            dump_var("skill_count", self._skill_counts)
            return self._current_plan_selection

        # ---- Global quota check ----
        if not self.should_attempt_injection_this_plan():
            self._current_plan_selection = None
            return None

        # ---- Geometric distribution decision: whether to inject this round ----
        if not self._should_inject_this_round(plan):
            self._current_plan_selection = None
            return None

        # ---- Extract candidate instances from the plan ----
        instances: List[Tuple[int, str, str]] = []
        skills_present: Dict[str, int] = {}

        for t_str, robots in (plan or {}).items():
            try:
                t = int(t_str)
            except Exception:
                continue
            for rlabel, spec in (robots or {}).items():
                skill = (spec or {}).get("skill")
                if not skill or skill in self._blacklist_skills:
                    continue
                instances.append((t, rlabel, skill))
                skills_present[skill] = skills_present.get(skill, 0) + 1

        if not instances:
            self._current_plan_selection = None
            return None

        # ---- Skill-balanced selection ----
        def _skill_cnt(s: str) -> int:
            idx = self._skill_to_idx.get(s)
            return 0 if idx is None else self._skill_counts[idx]

        min_cnt = min(_skill_cnt(s) for s in skills_present.keys())
        candidate_types = [s for s in skills_present.keys() if _skill_cnt(s) == min_cnt]
        chosen_type = self._rng.choice(candidate_types)

        # ---- Instance-balanced selection ----
        pool = [(t, rlabel, s) for (t, rlabel, s) in instances if s == chosen_type]
        sel = self._rng.choice(pool)
        self._current_plan_selection = sel

        dump_var("current_plan_selection", self._current_plan_selection)
        dump_var("skill_count", self._skill_counts)
        dump_var("injection_round", self._round_index)
        dump_var("injection_probability_info", {
            "round": self._round_index,
            "n_remaining": self.n_new_max - self.generated_count,
            "cooldown_remaining": self._cooldown_remaining,
            "rounds_since_last": self._rounds_since_last_injection,
        })
        return sel

    # =========================================================================
    # Dynamic Check Gating
    # =========================================================================

    def dynamic_check_enabled_for_instance(self, skill_name: str, timestep: int, robot_label: str) -> bool:
        if not self.should_attempt_injection_this_plan():
            return False
        return self._current_plan_selection == (int(timestep), robot_label, skill_name)

    def disable_all_dynamic_checks(self) -> bool:
        return not self.should_attempt_injection_this_plan()

    # =========================================================================
    # Count Backfill
    # =========================================================================

    def record_generated_for_skill(self, skill_name: str):
        """Call after successful injection: update counts, cooldown, and round log."""
        self.generated_count += 1

        # Skill count
        idx = self._skill_to_idx.get(skill_name)
        if idx is not None:
            self._skill_counts[idx] += 1

        # Reset cooldown and rounds since last injection
        self._cooldown_remaining = self.cooldown_rounds
        self._rounds_since_last_injection = 0

        # Record injection round
        self._round_injection_log.append(self._round_index)

    def record_generated_for_event(self, event_dict: Dict):
        op_code = event_dict.get("op_code")
        if not op_code:
            key = event_dict.get("event_key")
            if key:
                tpl = NEW_CASE_EVENT_TEMPLATES.get(key) or {}
                op_code = tpl.get("op_code")

        if op_code and op_code in self._op_to_idx:
            self._op_counts[self._op_to_idx[op_code]] += 1
        else:
            self._no_op_count += 1

    def get_op_usage_count(self, op_code: str) -> int:
        if not op_code:
            return 0
        idx = self._op_to_idx.get(op_code)
        return 0 if idx is None else self._op_counts[idx]

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self) -> Dict:
        return {
            "n_new_target": self.n_new_max,
            "generated_count": self.generated_count,
            "per_skill_counts": {"order": self._skill_order, "counts": list(self._skill_counts)},
            "per_newcase_type_counts": {
                "order": self._op_order + [self._no_op_bucket],
                "counts": list(self._op_counts) + [self._no_op_count],
            },
            "injection_strategy": "geometric_budget",
            "spacing_factor": self.spacing_factor,
            "cooldown_rounds": self.cooldown_rounds,
            "total_plan_rounds": self._round_index,
            "injection_round_log": list(self._round_injection_log),
        }
