# -*- coding: utf-8 -*-
"""
Action Converter - Converts natural language actions to PlanTranslateCoordinator's dispatcher_result structure.

Supports multiple matching strategies:
1. Exact pattern matching (regex) - supports multiple format variants
2. Semantic similarity matching (using sentence transformer)
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Natural-language action -> GSI skill_str mapping rules
# Regex supports: <>, spaces, underscores, and other separators
# ---------------------------------------------------------------------------

_ACTION_PATTERNS: List[tuple] = [
    # guide <target> to <location>
    # Matches: guide<target>_to<location>, guide <target> to <location>
    (re.compile(
        r"guide[<\s_]*(?P<target>[^>]+?)[\s>_]+to[<\s_]*(?P<location>[^>]+)[\s>]*$",
        re.IGNORECASE,
    ), "guide<{target}>_to<{location}>"),

    # place <object> on <surface>
    # Matches: place<object>_on<surface>, place <object> on <surface>
    (re.compile(
        r"place[<\s_]*(?P<object>[^>]+?)[\s>_]+on[<\s_]*(?P<surface>[^>]+)[\s>]*$",
        re.IGNORECASE,
    ), "place<{object}>_on<{surface}>"),

    # search <area> for <target>
    # Matches: search<area>_for<target>, search_area_for_target
    (re.compile(
        r"search[<\s_]*(?P<area>[^>]+?)[\s>_]+for[<\s_]*(?P<target>[^>]+)[\s>]*$",
        re.IGNORECASE,
    ), "search<{area}>_for<{target}>"),

    # navigate to <location> / navigate<location>
    (re.compile(
        r"navigate[<\s_]*(?:to\s+)?(?P<location>[^>]+)[\s>]*$",
        re.IGNORECASE,
    ), "navigate<{location}>"),

    # take photo of <target> / take_photo<target>
    (re.compile(
        r"take[\s_]photo[<\s_]*(?:of\s+)?(?P<target>[^>]+)[\s>]*$",
        re.IGNORECASE,
    ), "take_photo<{target}>"),

    # broadcast to <target> / broadcast<target>
    (re.compile(
        r"broadcast[<\s_]*(?:to\s+)?(?P<target>[^>]+)[\s>]*$",
        re.IGNORECASE,
    ), "broadcast<{target}>"),

    # follow <target>
    (re.compile(
        r"follow[<\s_]*(?P<target>[^>]+)[\s>]*$",
        re.IGNORECASE,
    ), "follow<{target}>"),

    # handle hazard <target> / handle_hazard<target>
    (re.compile(
        r"handle[\s_]hazard[<\s_]*(?P<target>[^>]+)[\s>]*$",
        re.IGNORECASE,
    ), "handle_hazard<{target}>"),

    # take off / take_off
    (re.compile(
        r"take[\s_]off[\s>]*$",
        re.IGNORECASE,
    ), "take_off"),

    # return home / return_home
    (re.compile(
        r"return[\s_]home[\s>]*$",
        re.IGNORECASE,
    ), "return_home"),

    # explicit idle / done / wait
    (re.compile(
        r"(?:stay\s+idle|done|wait|sync[\s_]wait)[\s>]*$",
        re.IGNORECASE,
    ), "sync_wait"),
]


class ActionConverter:
    """Converts natural language actions to GSI skill format.

    Core responsibilities:
    1. Maps each robot's natural language action string to ``skill_str``。
    2. Wraps the mapping result as a ``dispatcher_result`` dictionary.
    3. Supports multiple matching strategies and retry mechanisms.
    """

    def __init__(
        self, 
        step_prefix: str = "llamar_step",
        use_semantic_matching: bool = False,
        similarity_threshold: float = 0.7,
    ):
        """
        Args:
            step_prefix: Prefix used when generating task_id.
            use_semantic_matching: Whether to enable semantic similarity matching.
            similarity_threshold: Minimum similarity threshold for semantic matching.
        """
        self._step_prefix = step_prefix
        self._use_semantic_matching = use_semantic_matching
        self._similarity_threshold = similarity_threshold
        self._model = None
        self._skill_embeddings = None
        
        if use_semantic_matching:
            self._init_semantic_matcher()

    def _init_semantic_matcher(self):
        """Initialize semantic matching model (lazy loading)."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            from modules.task_solver.sgi_planner.prompt.atomic_skills import robot_skill_source
            
            all_skills = []
            for robot_type, skills_dict in robot_skill_source.items():
                for skill_data in skills_dict.values():
                    skill_name = skill_data.get("name", "")
                    if skill_name:
                        all_skills.append(skill_name)
            
            self._skill_templates = list(set(all_skills))
            self._skill_embeddings = torch.FloatTensor(
                self._model.encode(self._skill_templates)
            )
            
            logger.info(f"Semantic matcher initialized with {len(self._skill_templates)} skill templates")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic matcher: {e}")
            self._use_semantic_matching = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        robot_actions: Dict[str, str],
        step_num: int = 0,
    ) -> Tuple[Optional[Dict[str, Any]], bool, Dict[str, Tuple[str, float, str]]]:
        """Convert per-robot natural language actions to dispatcher_result format.

        Args:
            robot_actions: ``{robot_label: natural_language_action}`` dictionary.
            step_num: Current step number, used to generate task_id.

        Returns:
            (dispatcher_result, needs_retry, parse_details):
            - dispatcher_result: Dictionary compatible with PlanTranslateCoordinator, or None indicating complete failure
            - needs_retry: Whether retry is needed (some robot parsing failed)
            - parse_details: {robot_label: (parsed_action, confidence, original_action)}
        """
        task_id = f"{self._step_prefix}_{step_num}"
        skills: Dict[str, Dict[str, str]] = {}
        parse_details: Dict[str, Tuple[str, float, str]] = {}
        needs_retry = False
        
        for label, action_text in robot_actions.items():
            skill_str, confidence = self._parse_action_with_confidence(
                action_text.strip(), label
            )
            
            skills[label] = {
                "skill_str": skill_str, 
                "task_id": task_id,
                "parse_confidence": confidence,
            }
            
            parse_details[label] = (skill_str, confidence, action_text)
            
            # If fell back to sync_wait with low confidence, mark for retry
            if skill_str == "sync_wait" and confidence < 0.5:
                needs_retry = True

        # If all robots failed, return None
        if all(details[1] < 0.5 for details in parse_details.values()):
            return None, True, parse_details
        
        dispatcher_result = {"timestep_skills": {"0": skills}}
        return dispatcher_result, needs_retry, parse_details

    def parse_single_action(self, action_text: str) -> str:
        """Convert a single natural language action to GSI skill_str."""
        skill_str, _ = self._parse_action_with_confidence(action_text)
        return skill_str

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_action_with_confidence(
        self, action_text: str, robot_label: Optional[str] = None
    ) -> Tuple[str, float]:
        """Attempt to match a natural language action to GSI skill format, returning result and confidence.
        
        Returns:
            (skill_str, confidence): Skill string and confidence (0-1).
        """
        text = action_text.strip()
        if not text:
            return "sync_wait", 0.0

        # Strategy 1: Exact pattern matching
        for pattern, template in _ACTION_PATTERNS:
            m = pattern.match(text)
            if m:
                groups = {k: v.strip() for k, v in m.groupdict().items()}
                return template.format(**groups), 1.0

        # Strategy 2: Semantic similarity matching
        if self._use_semantic_matching and self._model is not None:
            skill_str, confidence = self._semantic_match(text)
            if confidence >= self._similarity_threshold:
                logger.info(
                    f"Semantic match for '{text}': {skill_str} (confidence: {confidence:.2f})"
                )
                return skill_str, confidence

        # Unrecognized -> fallback
        ctx = f" for robot {robot_label}" if robot_label else ""
        logger.warning(
            f"Unrecognized action{ctx}: '{text}'. Falling back to sync_wait."
        )
        return "sync_wait", 0.0

    def _semantic_match(self, action_text: str) -> Tuple[str, float]:
        """Use semantic similarity to match the closest skill.
        
        Returns:
            (skill_str, confidence): Best matching skill and similarity score.
        """
        try:
            import torch
            
            action_embedding = torch.FloatTensor(
                self._model.encode([action_text])
            )
            scores = torch.cosine_similarity(
                self._skill_embeddings, action_embedding
            )
            max_score, max_idx = torch.max(scores, 0)
            
            matched_skill = self._skill_templates[max_idx]
            confidence = float(max_score.item())
            
            return matched_skill, confidence
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return "sync_wait", 0.0
