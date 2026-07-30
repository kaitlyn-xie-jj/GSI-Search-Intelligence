# -*- coding: utf-8 -*-
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

from huggingface_hub import snapshot_download

# Try importing optional dependencies
try:
    from datasets import Dataset
except ImportError:
    raise ImportError("Please install dependencies: pip install datasets huggingface_hub")

# ==========================================
# Module Imports (Managers & Utils)
# ==========================================
from .managers.base import BaseDataManager
from .managers.scenario import ScenarioManager
from .managers.prompt import PromptManager

from .utils import (
    build_jsonl_index,
    read_jsonl_record,
    normalize_goal_data,
    extract_goal_metadata,
    load_dataset_from_file,
    detect_key_column,
)
from .splitter import DatasetSplitter


# ==========================================
# Default Local Dataset Path
# ==========================================
DEFAULT_LOCAL_PATH = Path(__file__).parent.parent.parent / "dataset"

# ==========================================
# Platform Type Constants
# ==========================================
VALID_PLATFORMS = {"semantic", "unreal"}


def _validate_platform_type(platform: str) -> None:
    """
    Validate platform type
    
    Args:
        platform: Platform type string
        
    Raises:
        ValueError: If platform type is invalid
    """
    if platform not in VALID_PLATFORMS:
        raise ValueError(
            f"Invalid platform type: {platform}. "
            f"Must be one of: {VALID_PLATFORMS}"
        )


def load_unreal_goal(
    goal_id: str,
    type_name: str = "cybertown",
    local_path: Optional[str] = None,
    lazy: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load goal data for Unreal platform

    Unreal platform only needs goals data; scene graph is obtained from UE5 in real-time.
    Provides simple goal loading interface without full task/scenario parsing.

    Args:
        goal_id: Goal ID (e.g., "g_48")
        type_name: Data type name (default "cybertown")
        local_path: Local dataset path, defaults to DEFAULT_LOCAL_PATH
        lazy: True=use file seek line-by-line search, False=load full dataset to memory

    Returns:
        Normalized goal data with instruction, goal_details, meta fields
        Returns None if not found
    """
    # Determine local dataset root directory
    base_path = Path(local_path) if local_path else DEFAULT_LOCAL_PATH

    # Build goals file path for unreal platform
    goals_file = base_path / "unreal" / "goals" / type_name / "goals.jsonl"

    if not goals_file.exists():
        raise FileNotFoundError(
            f"Unreal goals file not found: {goals_file}\n"
            f"Expected path: dataset/unreal/goals/{type_name}/goals.jsonl"
        )

    goal_data = None

    if lazy:
        # Lazy mode: scan file line-by-line to find target goal
        with open(goals_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    # Check id field match
                    record_id = record.get("id") or record.get("goal_id")
                    if str(record_id) == str(goal_id):
                        goal_data = record
                        break
                except json.JSONDecodeError:
                    continue
    else:
        # Full mode: load complete dataset to memory
        with open(goals_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get("id") or record.get("goal_id")
                    if str(record_id) == str(goal_id):
                        goal_data = record
                        break
                except json.JSONDecodeError:
                    continue

    if goal_data is None:
        return None

    # Normalize goal data
    return normalize_goal_data(goal_data)


class DatasetLoader:
    """
    Dataset Loader

    Responsibilities:
    1. **Resource Management**: Download and cache dataset files from HuggingFace Hub.
    2. **Component Coordination**: Manage four sub-managers: Tasks, Goals, Prompts, Scenarios.
    3. **Lazy Loading**: Support on-demand indexing and file reading, avoid loading large files to memory at once.
    4. **Dataset Splitting**: Provide stateful Splitter with non-overlapping splits, mixed sampling, K-Shot sampling.
    5. **Multi-platform Support**: Support data loading for semantic and unreal platforms.

    Attributes:
        repo_id (str): HuggingFace repository ID.
        type_name (str): Dataset type/subset name (default "cybertown").
        platform (str): Platform type ("semantic" or "unreal").
        token (str, optional): HF Access Token.
    """

    # Use module-level constant as default local dataset path
    DEFAULT_LOCAL_PATH = DEFAULT_LOCAL_PATH

    def __init__(
        self,
        repo_id: Optional[str] = None,
        type_name: str = "cybertown",
        platform: str = "semantic",
        token: Optional[str] = None,
        revision: str = "main",
        local_path: Optional[str] = None,
        use_local: bool = True,
    ):
        """
        Initialize loader

        Args:
            repo_id: Hugging Face Dataset repository ID (e.g., 'org/repo').
                     Can be None when use_local=True.
            type_name: Data subtype (corresponds to directory structure scenarios/<type_name>)
            platform: Platform type ("semantic" or "unreal"). Default "semantic".
                      semantic platform uses full dataset (goals, scenarios, prompts, tasks).
                      unreal platform only needs goals data.
            token: HF access token (required if not logged in via local CLI)
            revision: Repository branch or commit hash
            local_path: Local dataset root directory path. If None and use_local=True,
                        uses DEFAULT_LOCAL_PATH.
            use_local: Whether to use local mode. Default True, load directly from local;
                       if False, download from HuggingFace Hub.
        """
        # Validate platform type
        _validate_platform_type(platform)
        
        self.repo_id = repo_id
        self.type_name = type_name
        self.platform = platform
        self.token = token
        self.revision = revision
        self.use_local = use_local
        self.local_path = local_path

        # --- Sub-managers (Lazy Initialization) ---
        self._task_manager: Optional[BaseDataManager] = None
        self._goal_manager: Optional[BaseDataManager] = None
        self._prompt_manager: Optional[PromptManager] = None
        self._scenario_manager: Optional[ScenarioManager] = None

        # --- Cache & Index ---
        self._local_root: Optional[Path] = None  # Local download path
        self._platform_base_path: Optional[Path] = None  # Platform-specific base path
        self._task_ids_cache: Optional[List[str]] = None  # Task ID list cache
        self._file_indices: Dict[str, Dict[str, int]] = {}  # File byte offset index cache

        # --- Dataset Split State ---
        self._meta_index: Optional[List[Dict[str, Any]]] = None  # Metadata index (for filtering)
        self._splitter: Optional[DatasetSplitter] = None  # Internal Splitter instance

    # =========================================================================
    #  Core Resource Management
    # =========================================================================

    def _get_platform_base_path(self) -> Path:
        """
        Get platform-specific base path
        
        Returns correct data directory path based on platform type:
        - semantic platform: dataset/semantic/
        - unreal platform: dataset/unreal/
        
        Returns:
            Path: Platform-specific base path
        """
        self._ensure_local_root()
        
        if self._platform_base_path is not None:
            return self._platform_base_path
        
        if self.platform == "unreal":
            self._platform_base_path = self._local_root / "unreal"
        else:
            # semantic平台
            self._platform_base_path = self._local_root / "semantic"
        
        return self._platform_base_path

    def _ensure_local_root(self):
        """
        Ensure base file structure is available.

        - Local mode (use_local=True): Use local path directly, no download needed.
        - Remote mode (use_local=False): Use snapshot_download to download to local cache.
        """
        if self._local_root:
            return

        if self.use_local:
            # Local mode: use specified path or default path directly
            local_dir = Path(self.local_path or self.DEFAULT_LOCAL_PATH)
            if not local_dir.exists():
                raise FileNotFoundError(
                    f"Local dataset directory does not exist: {local_dir}\n"
                    f"Please check the path or set use_local=False to download from remote."
                )
            self._local_root = local_dir
        else:
            # Remote mode: download from HuggingFace Hub
            if not self.repo_id:
                raise ValueError(
                    "Remote mode requires repo_id. "
                    "Please set repo_id or use use_local=True."
                )
            
            # Select download mode based on platform type
            if self.platform == "unreal":
                allow_patterns = [
                    f"unreal/{self.type_name}/goals/goals.jsonl",
                ]
            else:
                # semantic platform: download full dataset; compatible with both new and old remote directory structures.
                allow_patterns = [
                    f"semantic/{self.type_name}/scenarios/*",
                    f"semantic/{self.type_name}/prompts/pool_*.json",
                    f"semantic/{self.type_name}/prompts/config.json",
                    f"semantic/{self.type_name}/prompts/prompts.jsonl",
                    f"semantic/{self.type_name}/prompts/prompts.jsonl.index",
                    f"semantic/{self.type_name}/tasks/tasks.jsonl",
                    f"semantic/{self.type_name}/tasks/tasks.jsonl.index",
                    f"semantic/{self.type_name}/goals/goals.jsonl",
                    f"semantic/{self.type_name}/goals/goals.jsonl.index",
                    f"scenarios/{self.type_name}/*",
                    f"prompts/{self.type_name}/pool_*.json",
                    f"prompts/{self.type_name}/config.json",
                    f"prompts/{self.type_name}/prompts.jsonl",
                    f"prompts/{self.type_name}/prompts.jsonl.index",
                    f"tasks/{self.type_name}/tasks.jsonl",
                    f"tasks/{self.type_name}/tasks.jsonl.index",
                    f"goals/{self.type_name}/goals.jsonl",
                    f"goals/{self.type_name}/goals.jsonl.index",
                ]
            
            self._local_root = Path(
                snapshot_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    allow_patterns=allow_patterns,
                    token=self.token,
                    revision=self.revision,
                )
            )

    def _resolve_data_path(self, data_type: str, filename: str) -> Path:
        """
        Resolve data file path, supports new and old directory structures
        
        New structure: dataset/{platform}/{data_type}/{type_name}/{filename}
        Old structure: dataset/{data_type}/{type_name}/{filename}
        
        Args:
            data_type: Data type (goals, tasks, prompts, scenarios)
            filename: Filename (e.g., goals.jsonl, tasks.jsonl)
            
        Returns:
            Path: Resolved file path
        """
        self._ensure_local_root()
        base_path = self._get_platform_base_path()
        
        # New structure path: dataset/{platform}/{data_type}/{type_name}/{filename}
        new_path = base_path / data_type / self.type_name / filename
        if new_path.exists():
            return new_path
        
        # Fallback to old structure: dataset/{data_type}/{type_name}/{filename}
        legacy_path = self._local_root / data_type / self.type_name / filename
        return legacy_path

    def _resolve_data_dir(self, data_type: str) -> Path:
        """
        Resolve data directory path, supports new and old directory structures
        
        New structure: dataset/{platform}/{data_type}/{type_name}/
        Old structure: dataset/{data_type}/{type_name}/
        
        Args:
            data_type: Data type (goals, tasks, prompts, scenarios)
            
        Returns:
            Path: Resolved directory path
        """
        self._ensure_local_root()
        base_path = self._get_platform_base_path()
        
        # New structure path
        new_dir = base_path / data_type / self.type_name
        if new_dir.exists():
            return new_dir
        
        # Fallback to old structure
        legacy_dir = self._local_root / data_type / self.type_name
        return legacy_dir

    def _get_record(
        self, file_path: Path, record_id: str, id_col: str
    ) -> Optional[Dict]:
        """
        Internal helper: combine index building and O(1) read.
        Delegates to utils and uses self._file_indices for caching.
        """
        index = build_jsonl_index(file_path, id_col, self._file_indices)
        return read_jsonl_record(file_path, record_id, index)

    def _build_metadata_index(self):
        """
        Build in-memory metadata index (Task ID -> Metadata).
        Index contains only lightweight fields needed for filtering (Level, Type, etc.) for Splitter,
        avoiding loading large Prompt text during filtering stage.
        """
        if self._meta_index is not None:
            return

        logging.info("Building Metadata Index for Splitting...")
        self._ensure_local_root()

        # 1. Load all Goal Metadata (lightweight extraction)
        # Use path resolution method to automatically handle new/old structures
        goals_file = self._resolve_data_path("goals", "goals.jsonl")
        
        goal_meta_map = {}

        if goals_file.exists():
            with open(goals_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        meta = extract_goal_metadata(record)
                        if meta["goal_id"]:
                            goal_meta_map[meta["goal_id"]] = meta
                    except Exception:
                        continue

        # 2. Iterate Tasks to establish associations (Task -> Goal ID -> Metadata)
        # Use path resolution method to automatically handle new/old structures
        tasks_file = self._resolve_data_path("tasks", "tasks.jsonl")
        
        task_index = []

        if tasks_file.exists():
            with open(tasks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        task = json.loads(line)
                        t_id = task.get("task_id")
                        g_id = task.get("goal")

                        # Merge Goal metadata into Task entry
                        if t_id and g_id and g_id in goal_meta_map:
                            entry = goal_meta_map[g_id].copy()
                            entry["task_id"] = t_id
                            entry["scenario_id"] = task.get("scenario")
                            task_index.append(entry)
                    except Exception:
                        continue

        self._meta_index = task_index
        logging.info(f"Metadata Index built: {len(self._meta_index)} tasks indexed.")

    # =========================================================================
    #  Advanced Splitting Interfaces (Splitter High-level Interfaces)
    # =========================================================================

    def get_stateful_splitter(
        self, seed: int = 42, reset: bool = False
    ) -> DatasetSplitter:
        """
        Get loader's internal stateful splitter.

        Features:
        1. Without reset, datasets obtained through this splitter are **strictly non-overlapping**.
        2. Suitable for sequential calls: get train, then val, then test.

        Args:
            seed: Random seed
            reset: Whether to force reset state (clear allocated records)
        """
        self._build_metadata_index()

        if self._splitter is None or reset:
            self._splitter = DatasetSplitter(self._meta_index, seed=seed)

        return self._splitter

    def get_subset(
        self,
        filters: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        limit: Optional[int] = None,
        ratio: Optional[float] = None,
        name: str = "subset",
        seed: int = 42,
        reset_splitter: bool = False,
    ) -> List[str]:
        """
        [Generic Interface] Get a dataset subset (Task IDs).
        Supports single filter, mixed filters, limit by count, limit by ratio.

        Args:
            filters:
                - Dict: Single filter (e.g. {"goal_type": "transport"})
                - List: Mixed filters (e.g. [{"filters":..., "weight":0.7}, ...])
                - None: No filter (take from all remaining data)
            limit: Fixed count (e.g. 1000). Higher priority than ratio.
            ratio: Ratio (e.g. 0.8). If limit is None, take ratio of available data. Default 1.0 (take all).
            name: Name for logging display.
            seed: Random seed.
            reset_splitter: Whether to reset state (if True, may overlap with previous history).

        Returns:
            List[str]: Selected task ID list
        """
        splitter = self.get_stateful_splitter(seed=seed, reset=reset_splitter)

        # Build specs list
        specs = []
        if isinstance(filters, list):
            # Already mixed specs (e.g. [{"filters": A, "weight": 0.5}, ...])
            specs = filters
        else:
            # Single condition
            # Default logic: if no mixed weight specified, treat as single group
            # If neither limit nor ratio specified, default take all (ratio=1.0)
            target_ratio = ratio if ratio is not None else 1.0
            specs = [{"filters": filters or {}, "ratio": target_ratio}]

        return splitter.split(specs, total_limit=limit, name=name)

    def get_train_test(
        self,
        train_filters: Optional[Dict[str, Any]] = None,
        test_filters: Optional[Dict[str, Any]] = None,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """
        [Shortcut Interface] Get non-overlapping Train / Test sets.
        This method automatically resets Splitter state to ensure splitting from full set.

        Example:
            >>> get_train_test(train_filters={"plan_level": "L2"}, test_filters={"plan_level": "L1"})
            >>> get_train_test(train_filters={"goal_type": "transport"}, train_ratio=0.8)

        Args:
            train_filters: Training set filter conditions
            test_filters: Test set filter conditions (if None, default take from remaining data)
            train_limit: Training set fixed count
            test_limit: Test set fixed count
            train_ratio: Training set ratio (effective when train_limit is None)
            seed: Random seed

        Returns:
            Dict: {"train": [id...], "test": [id...]}
        """
        # Force reset, start a new split cycle
        self.get_stateful_splitter(seed=seed, reset=True)

        # 1. Extract training set
        # If train_filters is None, means take from full set
        train_ids = self.get_subset(
            filters=train_filters,
            limit=train_limit,
            ratio=train_ratio if train_limit is None else 1.0,
            name="train",
        )

        # 2. Extract test set
        # When test_filters is None, default take all remaining data after train (ratio=1.0)
        # Or use user-specified test_filters
        # Note: reset_splitter=False here to ensure no overlap with train
        test_ids = self.get_subset(
            filters=(
                test_filters if test_filters is not None else train_filters
            ),  # If no test filter specified, default same source as train
            limit=test_limit,
            ratio=1.0,  # Default take all remaining (unless limit specified)
            name="test",
        )

        return {"train": train_ids, "test": test_ids}

    def get_k_shot(
        self, k: int, filters: Optional[Dict[str, Any]] = None, seed: int = 42
    ) -> List[str]:
        """
        [Shortcut Interface] Get K-Shot samples.
        Won't reset state, can call consecutively to get multiple non-overlapping shot groups.

        Args:
            k: Sample count
            filters: Filter conditions
            seed: Random seed
        """
        return self.get_subset(filters=filters, limit=k, seed=seed, name=f"{k}-shot")

    # =========================================================================
    #  Sub-Manager Accessors
    # =========================================================================

    @property
    def tasks(self) -> BaseDataManager:
        """Get Task manager"""
        if not self._task_manager:
            self._ensure_local_root()
            data_file = self._resolve_data_path("tasks", "tasks.jsonl")
            try:
                ds = load_dataset_from_file(data_file)
                self._task_manager = BaseDataManager(ds, key_column="task_id")
            except Exception:
                self._task_manager = BaseDataManager()
        return self._task_manager

    @property
    def goals(self) -> BaseDataManager:
        """Get Goal manager"""
        if not self._goal_manager:
            self._ensure_local_root()
            data_file = self._resolve_data_path("goals", "goals.jsonl")
            try:
                ds = load_dataset_from_file(data_file)
                # Auto-detect primary key
                key_column = detect_key_column(ds, ["id", "goal_id"])
                # Data integrity check
                if not key_column and len(ds) > 0:
                    first = ds[0]
                    if isinstance(first.get("goal_details"), str):
                        raise ValueError(
                            "No key column found. Data might need normalization."
                        )
                self._goal_manager = BaseDataManager(ds, key_column=key_column or "id")
            except Exception:
                raise Exception("Failed to load goals.jsonl")
        return self._goal_manager

    @property
    def prompts(self) -> PromptManager:
        """Get Prompt manager (Lazy Loading + Pools)"""
        if not self._prompt_manager:
            self._ensure_local_root()
            base_dir = self._resolve_data_dir("prompts")

            # 1. Load Pools
            pools = {}
            if base_dir.exists():
                for p_file in base_dir.glob("pool_*.json"):
                    try:
                        with open(p_file, "r", encoding="utf-8") as f:
                            pools[p_file.stem.replace("pool_", "")] = json.load(f)
                    except Exception:
                        continue

            # 2. Load Dataset
            data_file = base_dir / "prompts.jsonl"
            try:
                ds = load_dataset_from_file(data_file)
                self._prompt_manager = PromptManager(ds, pools)
            except Exception:
                self._prompt_manager = PromptManager(None, {})
        return self._prompt_manager

    @property
    def scenarios(self) -> ScenarioManager:
        """Get Scenario manager"""
        if not self._scenario_manager:
            self._ensure_local_root()
            # Check if new structure exists
            base_path = self._get_platform_base_path()
            new_scenarios_dir = base_path / "scenarios" / self.type_name
            
            if new_scenarios_dir.exists():
                # Use new structure: pass platform base path as root directory
                # ScenarioManager expects path format: root/scenarios/{type_name}/
                self._scenario_manager = ScenarioManager(base_path, self.type_name)
            else:
                # Fallback to old structure
                self._scenario_manager = ScenarioManager(self._local_root, self.type_name)
        return self._scenario_manager

    # =========================================================================
    #  Data Retrieval
    # =========================================================================

    def get_task(
        self,
        task_id: str,
        include_goal: bool = True,
        include_scenario: bool = False,
        include_prompt: bool = False,
        lazy: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete task information, automatically associate Goal, Scenario, Prompt.

        Args:
            task_id: Task ID
            include_goal: Whether to include goal info (normalized)
            include_scenario: Whether to include scene graph
            include_prompt: Whether to include restored Prompt
            lazy: True=slower but less memory, ~0.2s per record, False=read from memory Dataset (for training, ~0.02s per record)
        """

        # 1. Load Task basic info
        if lazy:
            self._ensure_local_root()
            path = self._resolve_data_path("tasks", "tasks.jsonl")
            task_info = self._get_record(path, task_id, "task_id")
        else:
            task_info = self.tasks.get_by_id(task_id)

        if not task_info:
            return None

        result = dict(task_info)
        # Structure task fields
        result["task"] = {
            "task_id": task_info.get("task_id"),
            "scenario": task_info.get("scenario"),
            "goal": task_info.get("goal"),
        }

        # 2. Associate Goal
        if include_goal and (gid := task_info.get("goal")):
            if lazy:
                path = self._resolve_data_path("goals", "goals.jsonl")
                goal_data = self._get_record(path, gid, "id")
            else:
                goal_data = self.goals.get_by_id(gid)

            if not goal_data:
                raise ValueError(f"Goal {gid} not found for task {task_id}")
            result["goal_details"] = normalize_goal_data(goal_data)

        # 3. Associate Scenario
        if include_scenario and (sid := task_info.get("scenario")):
            result["scene_graph"] = self.scenarios.get_scene_graph(sid)

        # 4. Associate Prompt
        if include_prompt:
            # PromptManager internally handles compression/decompression logic
            result["prompt_data"] = self.prompts.get_prompt(task_id)

        return result

    def list_task_ids(self, lazy: bool = True) -> List[str]:
        """
        List all task IDs.
        Uses index file for acceleration, avoids loading entire dataset.
        """
        if self._task_ids_cache:
            return self._task_ids_cache

        if lazy:
            self._ensure_local_root()
            path = self._resolve_data_path("tasks", "tasks.jsonl")
            index = build_jsonl_index(path, "task_id", self._file_indices)
            self._task_ids_cache = list(index.keys())
        else:
            if self.tasks.ds:
                self._task_ids_cache = list(self.tasks.ds["task_id"])
            else:
                self._task_ids_cache = []

        return self._task_ids_cache or []

    def refresh_cache(self):
        """Force reset all internal managers and caches"""
        self._task_manager = None
        self._goal_manager = None
        self._prompt_manager = None
        self._scenario_manager = None
        self._task_ids_cache = None
        self._meta_index = None
        self._splitter = None
        # self._file_indices not recommended to clear, as physical file offset generally doesn't change

