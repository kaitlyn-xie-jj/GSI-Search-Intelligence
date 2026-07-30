# Dataset Loader (`modules.dataset_loader`) 文档

## 概述

`modules.dataset_loader` 是 GSI 项目的统一数据集加载器，负责从**本地文件系统**或 **HuggingFace Hub** 加载并管理 GSI 数据集（Tasks、Goals、Prompts、Scenarios）。它支持懒加载、O(1) 索引检索、Prompt 去重解压，以及一套有状态的数据集划分 API。

### 主要特性

- 🚀 **懒加载 (Lazy Loading)**：按需初始化各子管理器，避免不必要的内存占用
- 📦 **统一接口**：一个 `DatasetLoader` 类协调所有数据类型
- 🌐 **多平台支持**：`semantic`（完整数据集）与 `unreal`（仅 goals，场景图由 UE5 实时提供）
- 💽 **本地优先**：默认从本地 `dataset/` 目录加载，必要时可切到 HuggingFace Hub
- 🔍 **快速检索**：基于 `.index` 文件 + 内存映射的 O(1) 数据查找
- 🧩 **去重 Prompt 自动解压**：从 `pool_*.json` 还原压缩的 prompt 记录
- ✂️ **数据集划分**：内建 `DatasetSplitter`，支持零重合切分、混合采样、K-Shot

## 模块结构

```
modules/dataset_loader/
├── __init__.py            # 暴露 DatasetLoader, load_unreal_goal
├── loader.py              # DatasetLoader 主类 + load_unreal_goal 函数
├── splitter.py            # DatasetSplitter（有状态划分器）
├── utils.py               # JSONL 索引、字段规范化、metadata 提取等工具
└── managers/
    ├── base.py            # BaseDataManager（HF Dataset 包装 + 索引）
    ├── prompt.py          # PromptManager（去重解压 + Prompt 拼装）
    └── scenario.py        # ScenarioManager（场景文件 LRU 读取）
```

## 安装

```bash
pip install datasets huggingface_hub
```

## 导入

```python
from modules.dataset_loader import DatasetLoader, load_unreal_goal
```

## 数据集目录约定

`DatasetLoader` 同时兼容两种目录结构，优先使用新结构、找不到时回退到旧结构。

**新结构（推荐）：**

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

**旧结构（仅作回退兼容）：**

```
dataset/
├── goals/<type_name>/goals.jsonl
├── tasks/<type_name>/tasks.jsonl
├── prompts/<type_name>/...
└── scenarios/<type_name>/...
```

`type_name` 默认为 `"cybertown"`。

## 快速开始

### 本地模式（默认）

```python
from modules.dataset_loader import DatasetLoader

# 默认 use_local=True，自动指向 <repo_root>/dataset
loader = DatasetLoader(type_name="cybertown", platform="semantic")

task_data = loader.get_task(
    task_id="cybertown_scenario_1_goal_1",
    include_goal=True,
    include_scenario=False,
    include_prompt=True,
    lazy=True,  # 按需读取，不加载整张表
)

task_ids = loader.list_task_ids(lazy=True)
```

### 远程模式（HuggingFace Hub）

```python
loader = DatasetLoader(
    repo_id="wenkangji/GSI",
    type_name="cybertown",
    platform="semantic",
    token=None,            # 私有仓库需要提供 HF token
    revision="main",
    use_local=False,       # 关闭本地模式，走 snapshot_download
)
```

### 训练场景（批量加载，单条更快）

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
        lazy=False,  # 命中已加载的内存数据集
    )
    # 训练逻辑...
```

### 可视化场景（按需读取，省内存）

```python
loader = DatasetLoader(type_name="cybertown")

def get_task_for_display(task_id: str):
    return loader.get_task(
        task_id=task_id,
        include_goal=True,
        include_scenario=True,
        include_prompt=True,
        lazy=True,  # 单条读取，磁盘 seek
    )
```

### Unreal 平台（仅 goals）

`unreal` 平台只需要 goals 数据，场景图由 UE5 在运行时提供。可使用快捷函数：

```python
from modules.dataset_loader import load_unreal_goal

goal = load_unreal_goal(goal_id="g_48", type_name="cybertown", lazy=True)
```

或通过 `DatasetLoader`：

```python
loader = DatasetLoader(platform="unreal", type_name="cybertown")
goal = loader.goals.get_by_id("g_48")
```

## 核心组件

### `DatasetLoader`

主加载器类，统一管理所有数据类型与划分逻辑。

#### 初始化参数

```python
DatasetLoader(
    repo_id: Optional[str] = None,   # HF 仓库 ID（仅远程模式必需）
    type_name: str = "cybertown",    # 数据子集名称
    platform: str = "semantic",      # "semantic" | "unreal"
    token: Optional[str] = None,     # HF Access Token（私有仓库）
    revision: str = "main",          # HF 分支或 commit
    local_path: Optional[str] = None,# 自定义本地数据根目录
    use_local: bool = True,          # True=本地，False=从 HF 下载
)
```

#### 数据检索方法

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

- `lazy=True`：通过 `.index` 文件做 O(1) seek 读取，单条约 0.2s，省内存。
- `lazy=False`：从已加载的 HuggingFace Dataset 读取，单条约 0.02s，更适合训练。

返回结构（字段视开关而定）：

```python
{
    "task_id": "...",
    "scenario": "...",
    "goal": "...",
    "task": {"task_id": "...", "scenario": "...", "goal": "..."},
    "goal_details": {...},   # include_goal=True 时
    "scene_graph": {...},    # include_scenario=True 时
    "prompt_data": {...},    # include_prompt=True 时（见 PromptManager.get_prompt）
}
```

##### `list_task_ids()`

```python
def list_task_ids(lazy: bool = True) -> List[str]
```

- `lazy=True`：只读 `.index`，不加载完整数据集。
- `lazy=False`：使用已加载的 `tasks.ds["task_id"]`。

##### `refresh_cache()`

重置所有内部 manager、`task_ids` 缓存、metadata 索引和 splitter 状态。文件 offset 索引（`_file_indices`）保留。

#### 子管理器属性（懒初始化）

| 属性 | 类型 | 说明 |
|------|------|------|
| `tasks` | `BaseDataManager` | 任务数据（`key_column="task_id"`） |
| `goals` | `BaseDataManager` | 目标数据（自动检测 `id` / `goal_id`） |
| `prompts` | `PromptManager` | Prompt 数据（自动加载 `pool_*.json` 并解压） |
| `scenarios` | `ScenarioManager` | 场景图与 plans（LRU 缓存） |

#### 数据集划分接口

`DatasetLoader` 内部持有一个有状态的 `DatasetSplitter`，负责构建轻量元数据索引（goal_type、plan_level、coor_level、language_level 等）并支持零重合划分。

##### `get_stateful_splitter()`

```python
def get_stateful_splitter(seed: int = 42, reset: bool = False) -> DatasetSplitter
```

返回内部维持的 splitter。多次调用之间不重置时，已分配的 task_id 不会再次出现。

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

- `filters` 为字典：单条件筛选，例如 `{"goal_type": "transport"}`。
- `filters` 为列表：混合采样，每项形如 `{"filters": {...}, "weight": 0.7}`。
- `limit` 优先于 `ratio`；都为空时默认 `ratio=1.0`（取剩余全部）。
- 支持的 level 字段（`plan_level` / `coor_level` / `language_level`）允许传入 lambda，例如 `lambda x: x < 3`。

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

强制重置 splitter，按筛选条件切出互不重合的 train / test。

##### `get_k_shot()`

```python
def get_k_shot(k: int, filters: Optional[Dict] = None, seed: int = 42) -> List[str]
```

不重置 splitter，可连续调用获取多组互不重合的 K-Shot 样本。

#### 划分示例

```python
# 1) 80/20 train/test
splits = loader.get_train_test(train_ratio=0.8)
train_ids, test_ids = splits["train"], splits["test"]

# 2) 按 goal_type 混合采样
ids = loader.get_subset(
    filters=[
        {"filters": {"goal_type": "transport"}, "weight": 0.7},
        {"filters": {"goal_type": "search"},    "weight": 0.3},
    ],
    limit=1000,
    name="mixed-1k",
)

# 3) Lambda 数值过滤
hard_ids = loader.get_subset(filters={"plan_level": lambda x: x >= 3}, ratio=1.0)

# 4) K-Shot
five_shot = loader.get_k_shot(k=5, filters={"goal_type": "transport"})
```

### `BaseDataManager`

`tasks` / `goals` 共用的轻量包装器，O(1) 检索基于内存索引。

```python
from modules.dataset_loader.managers.base import BaseDataManager

mgr: BaseDataManager = loader.tasks
record = mgr.get_by_id("cybertown_scenario_1_goal_1")
print(len(mgr))
```

| 方法/属性 | 说明 |
|-----------|------|
| `get_by_id(key)` | 通过主键获取记录 |
| `__len__()` | 数据集大小 |
| `ds` | 底层 HuggingFace Dataset |

### `PromptManager`

继承自 `BaseDataManager`，专门处理去重的 Prompt 数据（`prompts.jsonl` 中的字段是索引，需要从 `pool_*.json` 还原）。

#### `get_prompt()`

```python
def get_prompt(task_id: str) -> Dict[str, Any]
```

返回结构：

```python
{
    "task_id": "...",
    "type": "cybertown",
    "metadata": {...},                # 来自记录中的 metadata 字段
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
        "master_context": "...",      # 通过 runtime_builders.compose_master_context 生成
    },
    "prompt": "<完整拼装好的 prompt 字符串>",
}
```

> 注：内部会调用 `modules.task_solver.sgi_planner.prompt.runtime_builders.compose_master_context` 拼装 `master_context`。导入失败时返回占位符，不会抛异常。

### `ScenarioManager`

```python
sm = loader.scenarios
scene_graph = sm.get_scene_graph("scenario_1")  # 读 scenarios/<type_name>/scenario_1/scene_graph.json
plans       = sm.get_plans("scenario_1")        # 读 scenarios/<type_name>/scenario_1/plans.json
```

底层使用 `functools.lru_cache(maxsize=128)` 缓存 JSON 读取。

### `load_unreal_goal()`

模块级快捷函数，仅用于 `unreal/goals/<type_name>/goals.jsonl`：

```python
def load_unreal_goal(
    goal_id: str,
    type_name: str = "cybertown",
    local_path: Optional[str] = None,
    lazy: bool = True,
) -> Optional[Dict[str, Any]]
```

返回经过 `normalize_goal_data` 规范化后的 goal 记录；找不到时返回 `None`。

### `DatasetSplitter`

`splitter.py` 中独立可用，但建议通过 `loader.get_stateful_splitter()` 获取。核心方法：

```python
splitter.split(specs, total_limit=None, name="dataset") -> List[str]
splitter.reset()
```

- `_used_ids` 全局记录已分配的 task_id，保证多次 `split` 不重合。
- `specs` 支持 `dict` 或 `list[dict]`，每项可含 `filters`、`weight`、`ratio`。

## 数据流

### 资源准备

```
DatasetLoader(...) 实例化
        ↓
首次访问 tasks / goals / prompts / scenarios
        ↓
_ensure_local_root()
   ├─ use_local=True  → 直接定位 dataset/ 目录
   └─ use_local=False → snapshot_download 拉取需要的子集
        ↓
_resolve_data_path(...) 兼容新/旧目录结构
        ↓
load_dataset_from_file() 或 ScenarioManager 懒加载
        ↓
构建/复用索引（BaseDataManager / build_jsonl_index）
```

### Prompt 解压

```
prompts.jsonl 记录（含 *_idx 索引字段）
        ↓
PromptManager._inflate_and_format()
   ├─ 用 pool_*.json 把 idx 还原成文本片段
   ├─ compose_master_context(...) 组装 master_context
   └─ 用 head_template + response_format 拼出最终 prompt 字符串
        ↓
返回 {segments, metadata, prompt, ...}
```

### 索引文件 `*.jsonl.index`

`utils.build_jsonl_index` 优先读取同目录下的 `<file>.jsonl.index`（JSON 格式：`{id: byte_offset}`）。若不存在则实时扫描并填入内存缓存 `loader._file_indices`。

## 数据结构示例

### Task

```json
{
  "task_id": "cybertown_scenario_1_goal_1",
  "scenario": "scenario_1",
  "goal": "goal_1"
}
```

### Goal（规范化后）

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

### Prompt 记录（去重存储）

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

### Prompt 解压结果

见上文 [`PromptManager.get_prompt()`](#get_prompt)。

## 最佳实践

1. **训练流程**：用 `lazy=False` 一次性加载，再批量 `get_task` 或直接通过 `tasks/goals/prompts` 取用。
2. **可视化与调试**：用 `lazy=True`，按需 seek 单条记录，无需常驻内存。
3. **划分**：先调用 `get_train_test` / `get_subset`，再用返回的 `task_id` 列表去取数据；尽量复用同一个 `loader` 实例，让 splitter 维持零重合状态。
4. **远程 vs 本地**：默认本地读取最快；只有在 CI/容器场景需要直接拉数据集时才设 `use_local=False`。
5. **数据更新后**：调用 `loader.refresh_cache()` 清空 manager 与划分状态。

## 常见问题

**Q：本地数据放在哪里？**
默认是仓库根目录下的 `dataset/`（即 `Path(__file__).parent.parent.parent / "dataset"`）。可通过 `local_path` 参数自定义。

**Q：私有 HF 仓库怎么访问？**
设置 `use_local=False, repo_id="org/repo", token="hf_xxx"` 即可。

**Q：`goals.jsonl` 的主键是 `id` 还是 `goal_id`？**
`BaseDataManager` 会通过 `detect_key_column` 自动选择，两者都可。

**Q：`PromptManager.get_prompt()` 拿到的 `prompt` 字符串格式不对？**
确认能正确导入 `modules.task_solver.sgi_planner.prompt.runtime_builders`；该模块缺失时会回退到占位符。

**Q：`get_subset` / `get_train_test` 之间会不会重合？**
不会。`DatasetSplitter` 用 `_used_ids` 集合保障零重合，除非显式 `reset_splitter=True` 或调用 `get_train_test`（内部会强制 reset 一次再切）。

## API 速查

### `DatasetLoader`

| 成员 | 类型 | 说明 |
|------|------|------|
| `get_task()` | method | 获取单个任务及关联数据 |
| `list_task_ids()` | method | 列出所有 task_id |
| `refresh_cache()` | method | 重置 manager 与划分状态 |
| `get_stateful_splitter()` | method | 获取内部有状态 splitter |
| `get_subset()` | method | 通用子集采样（含混合权重） |
| `get_train_test()` | method | 切分零重合 train/test |
| `get_k_shot()` | method | 取 K-Shot 样本 |
| `tasks` | property → `BaseDataManager` | 任务管理器 |
| `goals` | property → `BaseDataManager` | 目标管理器 |
| `prompts` | property → `PromptManager` | Prompt 管理器 |
| `scenarios` | property → `ScenarioManager` | 场景管理器 |

### `BaseDataManager`

| 方法/属性 | 说明 |
|-----------|------|
| `get_by_id(key)` | 主键查询 |
| `__len__()` | 数据集大小 |
| `ds` | 底层 HuggingFace Dataset |

### `PromptManager`

| 方法 | 说明 |
|------|------|
| `get_prompt(task_id)` | 解压 + 拼装，返回含 `segments` / `prompt` 的字典 |

### `ScenarioManager`

| 方法 | 说明 |
|------|------|
| `get_scene_graph(scenario_id)` | 读 `scene_graph.json` |
| `get_plans(scenario_id)` | 读 `plans.json` |

### 模块级函数

| 函数 | 说明 |
|------|------|
| `load_unreal_goal(goal_id, ...)` | 直接读取 `unreal/goals/<type_name>/goals.jsonl` 中的单条 goal |

## 更新日志

### v3.0（当前）

- ✨ 新增 `platform` 参数，区分 `semantic` / `unreal` 两种数据布局
- ✨ 默认 `use_local=True`，从本地 `dataset/` 加载；保留 HF Hub 模式
- ✨ 新增数据集划分接口：`get_subset` / `get_train_test` / `get_k_shot` / `get_stateful_splitter`
- ✨ 新增 `load_unreal_goal` 快捷函数
- 🧩 `PromptManager.get_prompt()` 直接返回完整 prompt 字符串
- 🔧 兼容新旧两种目录结构：`dataset/{platform}/{data_type}/...` 与旧的 `dataset/{data_type}/...`

### v2.0

- 优化代码结构与统一加载方式
- 修复 instruction 提取逻辑

### v1.0

- 初始版本：懒加载、去重 prompt 解压
