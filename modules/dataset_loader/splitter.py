import random
import logging
from typing import List, Dict, Any, Optional, Union, Set


class DatasetSplitter:
    """
    Stateful Dataset Splitter

    Core Features:
    1. Zero Overlap: Maintains _used_ids set to ensure allocated tasks won't be selected again.
    2. Composition: Supports defining multiple rule groups, mixed generation by weight.
    3. Flexible Filtering: Supports Exact Match, List Inclusion, Lambda expressions.
    """

    def __init__(self, meta_index: List[Dict[str, Any]], seed: int = 42):
        """
        Args:
            meta_index: Task list containing metadata (built by Loader)
            seed: Random seed for reproducibility
        """
        self.meta_index = meta_index
        self._used_ids: Set[str] = set()  # State core: record allocated IDs
        self.rng = random.Random(seed)
        self.logger = logging.getLogger("DatasetSplitter")

        # Simple log config to prevent no handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def reset(self):
        """Reset state, clear allocated records (use with caution)"""
        self._used_ids.clear()
        self.logger.info("Splitter state reset. All IDs are available again.")

    def _match_filter(self, entry: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if single record matches Filter conditions
        Supports: string match, list inclusion match, Lambda function comparison
        """
        for key, cond in filters.items():
            # === Handle Level fields (numeric comparison support) ===
            if key in ["plan_level", "coor_level", "language_level"]:
                # Use utils pre-computed max_ value for numeric comparison
                max_val_key = f"max_{key}"
                entry_max = entry.get(max_val_key, -1)
                entry_raw = entry.get(key)

                # 1. Lambda/function comparison (e.g., lambda x: x < 3)
                if callable(cond):
                    if not cond(entry_max):
                        return False

                # 2. List/string inclusion match (e.g., cond="L1" or cond=["L1", "L2"])
                else:
                    # Unify to Set for intersection
                    entry_set = (
                        set(entry_raw) if isinstance(entry_raw, list) else {entry_raw}
                    )
                    req_set = set(cond) if isinstance(cond, list) else {cond}
                    if not (entry_set & req_set):
                        return False

            # === Handle regular fields (goal_type, task_id, etc.) ===
            else:
                entry_val = entry.get(key)
                if isinstance(cond, list):
                    if entry_val not in cond:
                        return False
                else:
                    if entry_val != cond:
                        return False
        return True

    def _query_candidates(self, filters: Dict[str, Any]) -> List[str]:
        """Query IDs that match conditions and are unused"""
        candidates = []
        for entry in self.meta_index:
            tid = entry["task_id"]
            # Key: skip allocated tasks
            if tid in self._used_ids:
                continue

            if self._match_filter(entry, filters):
                candidates.append(tid)
        return candidates

    def split(
        self,
        specs: Union[Dict[str, Any], List[Dict[str, Any]]],
        total_limit: Optional[int] = None,
        name: str = "dataset",
    ) -> List[str]:
        """
        Execute dataset split (core method)

        Args:
            specs: Sampling rule config.
                   Can be single dict: {"filters": {...}, "ratio": 0.8}
                   Or list (mixed mode):
                   [
                     {"filters": {"goal_type": "transport"}, "weight": 0.7},
                     {"filters": {"goal_type": "search"}, "weight": 0.3}
                   ]
            total_limit:
                   If integer specified (e.g., 1000), strictly limit total return count, allocate by weight.
                   If None, take ratio (default 1.0) of available data per specs.
            name: Dataset name for logging display.

        Returns:
            task_ids: Selected task ID list
        """
        if isinstance(specs, dict):
            specs = [specs]

        # 1. Preprocess weights (Normalize Weights)
        # If user didn't fill weight, default to 0, remaining quota evenly distributed later
        specified_weights = [s.get("weight", 0) for s in specs]
        total_spec_weight = sum(specified_weights)

        final_specs = []
        # Count how many items have unspecified weight
        undefined_count = len([w for w in specified_weights if w <= 0])

        # Remaining available weight (assume total weight normalized to 1.0)
        remaining_weight_quota = max(0.0, 1.0 - total_spec_weight)
        default_weight = (
            (remaining_weight_quota / undefined_count) if undefined_count > 0 else 0
        )

        for s in specs:
            w = s.get("weight", 0)
            if w <= 0:
                w = default_weight
            final_specs.append({**s, "_calc_weight": w})

        # 2. Execute stratified sampling
        result_ids = []

        for idx, spec in enumerate(final_specs):
            filters = spec.get("filters", {})
            candidates = self._query_candidates(filters)
            self.rng.shuffle(candidates)

            count_needed = 0

            # Strategy A: total_limit specified (e.g., want 1000, weight 0.7 -> target 700)
            if total_limit is not None:
                count_needed = int(total_limit * spec["_calc_weight"])
                # Handle rounding error for last group, ensure total aligns
                if idx == len(final_specs) - 1:
                    current_total = len(result_ids)
                    if current_total + count_needed < total_limit:
                        count_needed = total_limit - current_total

            # Strategy B: total_limit not specified, use internal ratio (e.g., take 80% of current remaining for this category)
            else:
                ratio = spec.get("ratio", 1.0)
                count_needed = int(len(candidates) * ratio)

            # Correct count (cannot exceed candidate total)
            actual_count = min(len(candidates), count_needed)

            selected = candidates[:actual_count]
            result_ids.extend(selected)

            # Mark as used, ensure no overlap later
            self._used_ids.update(selected)

            # Debug Log
            self.logger.info(
                f"[{name}] Segment {idx+1}: Filters={filters} | "
                f"Available={len(candidates)} | Wanted={count_needed} | Got={actual_count}"
            )

        # 3. Final shuffle
        self.rng.shuffle(result_ids)
        self.logger.info(f"[{name}] Final Output Size: {len(result_ids)}")

        return result_ids
