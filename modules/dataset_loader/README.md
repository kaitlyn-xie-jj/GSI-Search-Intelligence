# Dataset Loader (`modules.dataset_loader`) Documentation

## Overview

`modules.dataset_loader` is the unified dataset loader for the GSI project, responsible for loading and managing GSI datasets (Tasks, Goals, Prompts, Scenarios) from **local filesystem** or **HuggingFace Hub**. It supports lazy loading, O(1) indexed retrieval, Prompt deduplication/decompression, and a stateful dataset splitting API.

### Key Features

- 🚀 **Lazy Loading**: Initialize sub-managers on demand to avoid unnecessary memory usage
- 📦 **Unified Interface**: Single `DatasetLoader` class coordinates all data types
- 🌐 **Multi-platform Support**: `semantic` (full dataset) and `unreal` (goals only, scene graph from UE5)
- 💽 **Local-first**: Load from local `dataset/` directory by default, switch to HuggingFace Hub when needed
- 🔍 **Fast Retrieval**: O(1) data lookup based on `.index` files + memory mapping
- 🧩 **Auto Prompt Decompression**: Restore compressed prompt records from `pool_*.json`
- ✂️ **Dataset Splitting**: Built-in `DatasetSplitter` with zero-overlap splits, mixed sampling, K-Shot

## Module Structure

```
modules/dataset_loader/
├── __init__.py            # Exports DatasetLoader, load_unreal_goal
├── loader.py              # DatasetLoader main class + load_unreal_goal function
├── splitter.py            # DatasetSplitter (stateful splitter)
├── utils.py               # JSONL indexing, field normalization, metadata extraction tools
└── managers/
    ├── base.py            # BaseDataManager (HF Dataset wrapper + indexing)
    ├── prompt.py          # PromptManager (deduplication decompression + Prompt assembly)
    └── scenario.py        # ScenarioManager (scenario file LRU reading)
```

## Installation

```bash
pip install datasets huggingface_hub
```

## Import

```python
from modules.dataset_loader import DatasetLoader, load_unreal_goal
```

## Dataset Directory Convention

`DatasetLoader` supports both new and old directory structures, prioritizing new structure with fallback to old.

**New Structure (Recommended):**

```
dataset/
├── semantic/
│   ├── goals/<type_name>/goals.jsonl[.index]
│   ├── tasks/<type_name>/tasks.jsonl[.index]
│   ├── prompts/<type_name>/{prompts.jsonl[.index], pool_*.json, config.json}
│   └── scenarios/<type_name>/<scenario_id>/{scene_graph.json, plans.json}
└── unreal/
    └── goals/<type_name>/goals.jsonl
```

**Old Structure (Fallback Compatibility):**

```
dataset/
├── goals/<type_name>/goals.jsonl
├── tasks/<type_name>/tasks.jsonl
├── prompts/<type_name>/...
└── scenarios/<type_name>/...
```

`type_name` defaults to `"cybertown"`.

## Quick Start

### Local Mode (Default)

```python
from modules.dataset_loader import DatasetLoader

# Default use_local=True, auto-points to <repo_root>/dataset
loader = DatasetLoader(type_name="cybertown", platform="semantic")

task_data = loader.get_task(
    task_id="cybertown_scenario_1_goal_1",
    include_goal=True,
    include_scenario=False,
    include_prompt=True,
    lazy=True,  # On-demand read, doesn't load entire table
)

task_ids = loader.list_task_ids(lazy=True)
```

### Remote Mode (HuggingFace Hub)

```python
loader = DatasetLoader(
    repo_id="wenkangji/GSI",
    type_name="cybertown",
    platform="semantic",
    token=None,            # Private repos need HF token
    revision="main",
    use_local=False,       # Disable local mode, use snapshot_download
)
```

### Training Scenario (Batch Load, Faster per Item)

```python
loader = DatasetLoader(type_name="cybertown")

tasks = loader.tasks       # BaseDataManager
goals = loader.goals       # BaseDataManager
prompts = loader.prompts   # PromptManager

for task_id in loader.list_task_ids(lazy=False):
    task_data = loader.get_task(
        task_id=task_id,
        include_goal=True,
        include_prompt=True,
        lazy=False,  # Hit already-loaded in-memory dataset
    )
    # Training logic...
```

### Visualization Scenario (On-demand Read, Memory Efficient)

```python
loader = DatasetLoader(type_name="cybertown")

def get_task_for_display(task_id: str):
    return loader.get_task(
        task_id=task_id,
        include_goal=True,
        include_scenario=True,
        include_prompt=True,
        lazy=True,  # Single record read, disk seek
    )
```

### Unreal Platform (Goals Only)

`unreal` platform only needs goals data; scene graph provided by UE5 at runtime. Use shortcut function:

```python
from modules.dataset_loader import load_unreal_goal

goal = load_unreal_goal(goal_id="g_48", type_name="cybertown", lazy=True)
```

Or via `DatasetLoader`:

```python
loader = DatasetLoader(platform="unreal", type_name="cybertown")
goal = loader.goals.get_by_id("g_48")
```


## Core Components

### `DatasetLoader`

Main loader class that manages all data types and splitting logic.

#### Initialization Parameters

```python
DatasetLoader(
    repo_id: Optional[str] = None,   # HF repo ID (required for remote mode only)
    type_name: str = "cybertown",    # Dataset subset name
    platform: str = "semantic",      # "semantic" | "unreal"
    token: Optional[str] = None,     # HF Access Token (for private repos)
    revision: str = "main",          # HF branch or commit
    local_path: Optional[str] = None,# Custom local data root directory
    use_local: bool = True,          # True=local, False=download from HF
)
```

#### Data Retrieval Methods

##### `get_task()`

```python
def get_task(
    task_id: str,
    include_goal: bool = True,
    include_scenario: bool = False,
    include_prompt: bool = False,
    lazy: bool = True,
) -> Optional[Dict[str, Any]]
```

- `lazy=True`: O(1) seek read via `.index` file, ~0.2s per record, memory efficient.
- `lazy=False`: Read from loaded HuggingFace Dataset, ~0.02s per record, better for training.

Return structure (fields depend on flags):

```python
{
    "task_id": "...",
    "scenario": "...",
    "goal": "...",
    "task": {"task_id": "...", "scenario": "...", "goal": "..."},
    "goal_details": {...},   # When include_goal=True
    "scene_graph": {...},    # When include_scenario=True
    "prompt_data": {...},    # When include_prompt=True (see PromptManager.get_prompt)
}
```

##### `list_task_ids()`

```python
def list_task_ids(lazy: bool = True) -> List[str]
```

- `lazy=True`: Only read `.index`, don't load full dataset.
- `lazy=False`: Use loaded `tasks.ds["task_id"]`.

##### `refresh_cache()`

Reset all internal managers, `task_ids` cache, metadata index, and splitter state. File offset index (`_file_indices`) is preserved.

#### Sub-Manager Properties (Lazy Initialization)

| Property | Type | Description |
|------|------|------|
| `tasks` | `BaseDataManager` | Task data (`key_column="task_id"`) |
| `goals` | `BaseDataManager` | Goal data (auto-detect `id` / `goal_id`) |
| `prompts` | `PromptManager` | Prompt data (auto-load `pool_*.json` and decompress) |
| `scenarios` | `ScenarioManager` | Scene graphs and plans (LRU cached) |

#### Dataset Splitting Interface

`DatasetLoader` maintains an internal stateful `DatasetSplitter` that builds lightweight metadata index (goal_type, plan_level, coor_level, language_level, etc.) and supports zero-overlap splits.

##### `get_stateful_splitter()`

```python
def get_stateful_splitter(seed: int = 42, reset: bool = False) -> DatasetSplitter
```

Returns internal splitter. Without reset, allocated task_ids won't appear again across multiple calls.

##### `get_subset()`

```python
def get_subset(
    filters: Optional[Union[Dict, List[Dict]]] = None,
    limit: Optional[int] = None,
    ratio: Optional[float] = None,
    name: str = "subset",
    seed: int = 42,
    reset_splitter: bool = False,
) -> List[str]
```

- `filters` as dict: Single filter, e.g., `{"goal_type": "transport"}`.
- `filters` as list: Mixed sampling, each item like `{"filters": {...}, "weight": 0.7}`.
- `limit` takes priority over `ratio`; both empty defaults to `ratio=1.0` (take all remaining).
- Level fields (`plan_level` / `coor_level` / `language_level`) support lambda, e.g., `lambda x: x < 3`.

##### `get_train_test()`

```python
def get_train_test(
    train_filters: Optional[Dict] = None,
    test_filters: Optional[Dict] = None,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[str]]
```

Force reset splitter, split non-overlapping train/test by filter conditions.

##### `get_k_shot()`

```python
def get_k_shot(k: int, filters: Optional[Dict] = None, seed: int = 42) -> List[str]
```

Don't reset splitter, can call consecutively to get multiple non-overlapping K-Shot sample groups.

#### Splitting Examples

```python
# 1) 80/20 train/test
splits = loader.get_train_test(train_ratio=0.8)
train_ids, test_ids = splits["train"], splits["test"]

# 2) Mixed sampling by goal_type
ids = loader.get_subset(
    filters=[
        {"filters": {"goal_type": "transport"}, "weight": 0.7},
        {"filters": {"goal_type": "search"},    "weight": 0.3},
    ],
    limit=1000,
    name="mixed-1k",
)

# 3) Lambda numeric filtering
hard_ids = loader.get_subset(filters={"plan_level": lambda x: x >= 3}, ratio=1.0)

# 4) K-Shot
five_shot = loader.get_k_shot(k=5, filters={"goal_type": "transport"})
```

### `BaseDataManager`

Lightweight wrapper shared by `tasks` / `goals`, O(1) retrieval based on memory index.

```python
from modules.dataset_loader.managers.base import BaseDataManager

mgr: BaseDataManager = loader.tasks
record = mgr.get_by_id("cybertown_scenario_1_goal_1")
print(len(mgr))
```

| Method/Property | Description |
|-----------|------|
| `get_by_id(key)` | Get record by primary key |
| `__len__()` | Dataset size |
| `ds` | Underlying HuggingFace Dataset |

### `PromptManager`

Inherits from `BaseDataManager`, specifically handles deduplicated Prompt data (`prompts.jsonl` fields are indices, need restoration from `pool_*.json`).

#### `get_prompt()`

```python
def get_prompt(task_id: str) -> Dict[str, Any]
```

Return structure:

```python
{
    "task_id": "...",
    "type": "cybertown",
    "metadata": {...},                # From record's metadata field
    "segments": {
        "skill_set": "...",
        "env_desc": "...",
        "goal_notes": "...",
        "core_def": "...",
        "univ_rules": "...",
        "available_robots": "...",
        "response_format": "...",
        "head_template": "...",
        "instruction": "...",
        "feedback_context": "...",
        "master_context": "...",      # Generated via runtime_builders.compose_master_context
    },
    "prompt": "<fully assembled prompt string>",
}
```

> Note: Internally calls `modules.task_solver.sgi_planner.prompt.runtime_builders.compose_master_context` to assemble `master_context`. Returns placeholder on import failure without throwing exception.

### `ScenarioManager`

```python
sm = loader.scenarios
scene_graph = sm.get_scene_graph("scenario_1")  # Read scenarios/<type_name>/scenario_1/scene_graph.json
plans       = sm.get_plans("scenario_1")        # Read scenarios/<type_name>/scenario_1/plans.json
```

Uses `functools.lru_cache(maxsize=128)` to cache JSON reads.

### `load_unreal_goal()`

Module-level shortcut function for `unreal/goals/<type_name>/goals.jsonl` only:

```python
def load_unreal_goal(
    goal_id: str,
    type_name: str = "cybertown",
    local_path: Optional[str] = None,
    lazy: bool = True,
) -> Optional[Dict[str, Any]]
```

Returns goal record normalized via `normalize_goal_data`; returns `None` if not found.

### `DatasetSplitter`

Independently usable in `splitter.py`, but recommended to get via `loader.get_stateful_splitter()`. Core method:

```python
splitter.split(specs, total_limit=None, name="dataset") -> List[str]
splitter.reset()
```

- `_used_ids` globally tracks allocated task_ids, ensuring multiple `split` calls don't overlap.
- `specs` supports `dict` or `list[dict]`, each item can contain `filters`, `weight`, `ratio`.


## Data Flow

### Resource Preparation

```
DatasetLoader(...) instantiation
        ↓
First access to tasks / goals / prompts / scenarios
        ↓
_ensure_local_root()
   ├─ use_local=True  → Directly locate dataset/ directory
   └─ use_local=False → snapshot_download pulls needed subset
        ↓
_resolve_data_path(...) compatible with new/old directory structures
        ↓
load_dataset_from_file() or ScenarioManager lazy load
        ↓
Build/reuse index (BaseDataManager / build_jsonl_index)
```

### Prompt Decompression

```
prompts.jsonl record (contains *_idx index fields)
        ↓
PromptManager._inflate_and_format()
   ├─ Use pool_*.json to restore idx to text segments
   ├─ compose_master_context(...) assemble master_context
   └─ Use head_template + response_format to build final prompt string
        ↓
Return {segments, metadata, prompt, ...}
```

### Index Files `*.jsonl.index`

`utils.build_jsonl_index` prioritizes reading `<file>.jsonl.index` in same directory (JSON format: `{id: byte_offset}`). If not exists, scans in real-time and fills memory cache `loader._file_indices`.

## Data Structure Examples

### Task

```json
{
  "task_id": "cybertown_scenario_1_goal_1",
  "scenario": "scenario_1",
  "goal": "goal_1"
}
```

### Goal (Normalized)

```json
{
  "id": "goal_1",
  "instruction": "Search for a red car in the garden",
  "goal_details": {
    "goal_id": "goal_1",
    "goal_type": "area_search",
    "description": "Search for a red car in the garden",
    "core_params": {}
  },
  "meta": {
    "language_level": "L1",
    "plan_level": ["L1", "L2"],
    "coor_level": ["L0"]
  }
}
```

### Prompt Record (Deduplicated Storage)

```json
{
  "task_id": "cybertown_scenario_1_goal_1",
  "skill_set_idx": 0,
  "env_desc_idx": 1,
  "available_robots_idx": 2,
  "goal_notes_idx": 4,
  "core_def_idx": 0,
  "univ_rules_idx": 0,
  "head_template_idx": 0,
  "response_format_idx": 0,
  "instruction": "Search for a red car in the garden",
  "feedback_context": "",
  "metadata": {
    "goal_id": "goal_1",
    "scenario_id": "scenario_1",
    "goal_type": "area_search"
  }
}
```

### Prompt Decompression Result

See [`PromptManager.get_prompt()`](#get_prompt) above.

## Best Practices

1. **Training Pipeline**: Use `lazy=False` to load once, then batch `get_task` or directly use `tasks/goals/prompts`.
2. **Visualization & Debugging**: Use `lazy=True` to seek single records on demand, no memory resident.
3. **Splitting**: Call `get_train_test` / `get_subset` first, then use returned `task_id` list to fetch data; reuse same `loader` instance to let splitter maintain zero-overlap state.
4. **Remote vs Local**: Default local read is fastest; only set `use_local=False` when CI/container needs direct dataset pull.
5. **After Data Update**: Call `loader.refresh_cache()` to clear manager and split state.

## FAQ

**Q: Where to put local data?**
Default is `dataset/` under repo root (i.e., `Path(__file__).parent.parent.parent / "dataset"`). Customize via `local_path` parameter.

**Q: How to access private HF repos?**
Set `use_local=False, repo_id="org/repo", token="hf_xxx"`.

**Q: Is `goals.jsonl` primary key `id` or `goal_id`?**
`BaseDataManager` auto-selects via `detect_key_column`, both work.

**Q: `PromptManager.get_prompt()` returned `prompt` string format incorrect?**
Confirm correct import of `modules.task_solver.sgi_planner.prompt.runtime_builders`; falls back to placeholder if module missing.

**Q: Will `get_subset` / `get_train_test` overlap?**
No. `DatasetSplitter` uses `_used_ids` set to ensure zero overlap, unless explicitly `reset_splitter=True` or call `get_train_test` (internally forces reset once before split).

## API Quick Reference

### `DatasetLoader`

| Member | Type | Description |
|------|------|------|
| `get_task()` | method | Get single task with associated data |
| `list_task_ids()` | method | List all task_id |
| `refresh_cache()` | method | Reset manager and split state |
| `get_stateful_splitter()` | method | Get internal stateful splitter |
| `get_subset()` | method | Generic subset sampling (with mixed weights) |
| `get_train_test()` | method | Split zero-overlap train/test |
| `get_k_shot()` | method | Get K-Shot samples |
| `tasks` | property → `BaseDataManager` | Task manager |
| `goals` | property → `BaseDataManager` | Goal manager |
| `prompts` | property → `PromptManager` | Prompt manager |
| `scenarios` | property → `ScenarioManager` | Scenario manager |

### `BaseDataManager`

| Method/Property | Description |
|-----------|------|
| `get_by_id(key)` | Primary key query |
| `__len__()` | Dataset size |
| `ds` | Underlying HuggingFace Dataset |

### `PromptManager`

| Method | Description |
|------|------|
| `get_prompt(task_id)` | Decompress + assemble, return dict with `segments` / `prompt` |

### `ScenarioManager`

| Method | Description |
|------|------|
| `get_scene_graph(scenario_id)` | Read `scene_graph.json` |
| `get_plans(scenario_id)` | Read `plans.json` |

### Module-level Functions

| Function | Description |
|------|------|
| `load_unreal_goal(goal_id, ...)` | Directly read single goal from `unreal/goals/<type_name>/goals.jsonl` |

## Changelog

### v3.0 (Current)

- ✨ Added `platform` parameter to distinguish `semantic` / `unreal` data layouts
- ✨ Default `use_local=True`, load from local `dataset/`; retain HF Hub mode
- ✨ Added dataset splitting interfaces: `get_subset` / `get_train_test` / `get_k_shot` / `get_stateful_splitter`
- ✨ Added `load_unreal_goal` shortcut function
- 🧩 `PromptManager.get_prompt()` directly returns complete prompt string
- 🔧 Compatible with new and old directory structures: `dataset/{platform}/{data_type}/...` and old `dataset/{data_type}/...`

### v2.0

- Optimized code structure and unified loading approach
- Fixed instruction extraction logic

### v1.0

- Initial version: lazy loading, deduplicated prompt decompression
