# 运行时配置说明

本文档描述 `config/default.json` 中的核心可配参数。

---

## 规划器设置

### `planner_mode`
- `"full"`：一次性生成完整计划，包括参数不明确的任务。
- `"phase"`：仅生成参数明确的阶段计划（已弃用）。

### `default_robot_types`
系统默认支持的机器人类型列表。

---

## 运行时 / 调试

### `enable_detailed_print`
启用更详细的日志输出（调试用）。

### `simulate_time_delay`
是否模拟技能执行耗时。批量测试时建议设为 `false`。

### `enable_visualization`
是否启用可视化窗口。

### `enable_video_recording`
是否录制执行视频。需先启用可视化才能录制。

### `fine_grained_simulation`
是否启用细粒度仿真步，决定是否启用轻量物理仿真。建议开启。

### `max_concurrency`
最大并发执行数。

### `enable_logging`
是否保存运行时日志。

### `enable_checkpointing`
是否在关键步骤保存状态快照（用于恢复/回放）。

---

## 重规划与数据集采集

### `enable_replanning`
是否启用重规划功能。启用后，系统在突发事件或执行失败时尝试重新规划。

### `collect_replan_dataset`
是否采集重规划场景数据集（用于微调）。默认禁用。若启用，需同时启用 `enable_new_case_generation`，并将 `max_newcases_per_run` 设为至少 1。

---

## 突发事件生成

### `enable_new_case_generation`
是否启用突发事件注入。

### `max_newcases_per_run`
每次任务执行中允许的最大突发事件数。仅在 `enable_new_case_generation` 启用时生效。

### `new_case_mode`
- 仅在 `enable_new_case_generation` 启用时生效。
- `"aggregate"`：部分策略采用"非回滚"聚合处理，即不恢复之前的图操作。预检在所有条件验证后统一报告失败事件。
- `"immediate"`：使用原始模板回滚行为执行，即恢复之前的图操作。预检在任一条件失败时立即报告。

### `newcase_spacing_factor`
几何分布参数，控制突发事件注入间隔。值越大注入越稀疏。系统通过 `window = 剩余可注入次数 × spacing_factor` 计算动态窗口，再以 `p = 剩余次数 / window` 作为当前轮注入概率。最小值 `1.0`，对应每轮最高注入概率。仅在 `enable_new_case_generation` 启用时生效。

### `newcase_cooldown_rounds`
每次成功注入后的冷却轮数。冷却期内注入概率强制为 `0`，防止连续密集注入。设为 `0` 表示无冷却。仅在 `enable_new_case_generation` 启用时生效。

### `newcase_similarity_threshold`
相邻轮次计划间的 Jaccard 相似度阈值。当连续计划的技能指纹相似度超过此阈值时，系统认为计划变化极小，对注入概率施加衰减（乘以 `newcase_similarity_damping`），避免在高度相似的计划上重复注入。取值范围 `[0, 1]`，越低越容易触发衰减。仅在 `enable_new_case_generation` 启用时生效。

### `newcase_similarity_damping`
当计划相似度超过 `newcase_similarity_threshold` 时，注入概率的衰减系数。例如设为 `0.3` 表示概率降至原来的 30%。取值范围 `[0, 1]`，越低衰减越强。仅在 `enable_new_case_generation` 启用时生效。

---

## 数据集与平台

### `repo_id`
数据集仓库 ID，用于上传/下载数据集时标识远程仓库地址。

### `platform_type`
运行平台类型。`"semantic"` 为语义仿真平台（轻量），`"unreal"` 为 Unreal Engine 仿真平台。

### `unreal_platform`
Unreal Engine 平台连接配置。仅在 `platform_type` 为 `"unreal"` 时生效。
- `base_url`：UE5 仿真服务的 HTTP 地址。
- `timeout`：单次请求超时（秒）。
- `polling_interval`：执行状态轮询间隔（秒）。

---

## 回放模式

用于回放已有轨迹，不执行实际规划过程。

### `replay_mode.enabled`
是否启用回放模式。启用后可完整复现一次先前的实验。

### `replay_mode.trace_root`
回放轨迹文件根目录，即包含 `temp_var.json` 的目录。仅在 `replay_mode.enabled` 启用时生效。

### `replay_mode.trace_tag`
回放标签（用于区分多条轨迹）。仅在 `replay_mode.enabled` 启用时生效。

---

## 人机协同（HITL）

人机协同交互配置，用于在执行过程中引入人类操作员的指令输入、计划审阅和决策干预。

### `human_in_loop.enabled`
HITL 总开关。设为 `false` 时以下所有子功能均禁用，系统完全自主运行。

### `human_in_loop.instruction_enabled`
是否启用指令输入功能。启用后，系统在任务开始前请求操作员通过 UE5 输入指令。需 `enabled` 为 `true`。

### `human_in_loop.review_enabled`
是否启用计划审阅功能。启用后，系统生成的计划在执行前发送给操作员审阅和修改。需 `enabled` 为 `true`。

### `human_in_loop.decision_enabled`
是否启用决策请求功能。启用后，系统在遇到歧义或需要人工判断时（如搜索目标未找到）请求操作员决策。需 `enabled` 为 `true`。

### `human_in_loop.instruction_timeout`
等待操作员指令输入的超时时间（秒）。超时后系统使用预加载指令继续。

### `human_in_loop.review_timeout`
等待操作员完成计划审阅的超时时间（秒）。超时后系统使用原始计划继续。

### `human_in_loop.decision_timeout`
等待操作员决策的超时时间（秒）。超时后系统使用默认决策（`end_task`）继续。

### `human_in_loop.server_port`
HITL 消息通信用的 Python 端 HTTP 服务端口。

### `human_in_loop.retry_count`
HITL 操作失败时的重试次数。

### `human_in_loop.retry_delay`
重试间隔（秒）。

---

## 求解器配置

### `solver_type`
当前使用的求解器类型。可选：`"sgi"`、`"llamar"`、`"spine"`、`"lipllm"`、`"smartllm"`。系统根据此值从 `solver_config` 中读取对应配置块。

### `solver_config`
各求解器的独立配置，按求解器名称索引。

#### 通用参数（所有求解器支持）

| 参数 | 说明 |
|------|------|
| `max_steps` | 最大执行步数（规划-执行循环轮数上限） |
| `model_family` | 使用的 LLM 模型族。`null` 时使用系统默认模型 |
| `model_name_override` | 模型名称覆盖。`null` 时使用 `model_family` 的默认模型 |

#### `sgi` 专有参数

| 参数 | 说明 |
|------|------|
| `validate_plan` | 是否验证生成的计划 |

#### `llamar` 专有参数

| 参数 | 说明 |
|------|------|
| `use_few_shot` | 是否在 prompt 中使用 few-shot 示例引导 LLM 生成 |

#### `spine` 专有参数

| 参数 | 说明 |
|------|------|
| `use_few_shot` | 是否使用 few-shot 示例 |
| `n_attempts` | 每轮规划的最大尝试次数，生成失败时重试 |

#### `lipllm` 专有参数

| 参数 | 说明 |
|------|------|
| `use_few_shot` | 是否使用 few-shot 示例 |
| `n_attempts` | 每轮规划的最大尝试次数 |
| `max_iterations` | LipLLM 内部迭代优化的最大轮数 |
| `alpha` | 迭代优化中的步长/权重系数，控制每轮更新幅度 |

#### `smartllm` 专有参数

| 参数 | 说明 |
|------|------|
| `use_few_shot` | 是否使用 few-shot 示例 |

---

扩展或具体用法详情请参考各模块的实现文档。
