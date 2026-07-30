# 配置说明

`config/default.json` 定义 GSI 运行时默认行为，包括 planner、平台、replan、newcase、HITL 和部分 solver 配置。模型服务地址、Hugging Face cache、benchmark 数据集路径和 TANGO 后端通常通过 CLI 参数或环境变量设置。

## 配置来源

常见优先级：

1. 运行脚本的 CLI 参数。
2. 显式环境变量。
3. `config/default.json`。
4. 脚本或模块默认值。

常见入口：

- `python run/run_exp_multi_method.py ...`
- `python run/run_collect_replan_dataset.py ...`
- `config/llm_finetune.json`

## 推荐默认思路

```json
{
  "planner_mode": "full",
  "platform_type": "semantic",
  "solver_type": "sgi",
  "enable_replanning": true,
  "enable_new_case_generation": false,
  "collect_replan_dataset": false
}
```

默认使用 SGI、semantic platform 和重规划；普通 benchmark 不主动注入 new case。New case 评估和 replan 数据采集应通过脚本参数显式开启。

## Planner 与 LLM

- `planner_mode=full`：推荐模式，一次生成完整任务计划。
- `planner_mode=phase`：旧阶段式模式，不作为默认推荐路径。
- `llm_temperature`：评估和数据采集通常设为 `0.0`。
- `default_robot_types`：参与规划的机器人类型白名单。

默认机器人类型：

```json
["UAV", "UGV", "Quadruped", "Humanoid"]
```

## Runtime 与调试

- `enable_detailed_print`：输出更详细的运行日志，适合单例调试。
- `simulate_time_delay`：是否模拟技能耗时；批量评估通常关闭。
- `enable_visualization`：是否启用可视化。
- `enable_video_recording`：是否录制视频。
- `fine_grained_simulation`：是否启用细粒度语义仿真。
- `max_concurrency`：默认并发上限；批处理脚本通常由 `--max-workers` 覆盖。
- `enable_logging`、`enable_checkpointing`：控制日志和中间状态记录。

## Replan 与 New Case

- `enable_replanning`：重规划总开关。
- `enable_new_case_generation`：new case 注入开关。
- `collect_replan_dataset`：是否采集 replan 训练数据。
- `max_newcases_per_run`：单个 run 的 new case 上限。
- `new_case_mode`：new case 处理策略。
- `newcase_spacing_factor`：注入间隔控制。
- `newcase_cooldown_rounds`：注入后的冷却轮数。
- `newcase_similarity_threshold`、`newcase_similarity_damping`：相似计划下的重复注入抑制。

New case benchmark 示例：

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 4 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/eval_multi_method/qwen3_0_6b_cybertown
```

## 数据集与平台

`repo_id` 是默认远程任务数据集仓库，当前为：

```text
WindyLab/GSI
```

正式运行建议显式设置数据集路径：

```bash
export GSI_DATASET_ROOT=$(
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="WindyLab/GSI",
    repo_type="dataset",
    revision="small",
))
PY
)
```

平台类型：

- `semantic`：默认平台，适合训练、validator、benchmark 和批量评估。
- `unreal`：连接 Unreal Engine 服务，适合 UE5 联动。

`platform_type=unreal` 时，`unreal_platform` 中的 `base_url`、`timeout` 和 `polling_interval` 才会生效。

## Replay Mode

Replay mode 用于回放已有轨迹，而不是重新请求 LLM：

- `enabled`：总开关。
- `trace_root`：已有 trace 根目录。
- `trace_tag`：同一目录下的 trace 标签。

普通评估和训练后验证应保持 `replay_mode.enabled=false`。

## Human-In-The-Loop

HITL 用于人工介入流程，主要服务 Unreal 或交互式运行。非交互 benchmark 通常关闭：

- `enabled`
- `instruction_enabled`
- `review_enabled`
- `decision_enabled`
- `instruction_timeout`
- `review_timeout`
- `decision_timeout`
- `server_port`
- `retry_count`
- `retry_delay`

## Planner 方法

`solver_type` 选择 planner 方法，不是 TANGO 优化后端。常见值：

- `sgi`
- `llamar`
- `spine`
- `lipllm`
- `smartllm`

`solver_config` 保存各方法参数。常见字段：

- `max_steps`
- `model_family`
- `model_name_override`
- `validate_plan`
- `use_few_shot`
- `n_attempts`
- `max_iterations`
- `alpha`

多方法 benchmark 应使用 CLI：

```bash
python run/run_exp_multi_method.py \
  --methods sgi spine smartllm lipllm \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/eval_multi_method/qwen3_0_6b_cybertown_baselines
```

## 不写入 JSON 的配置

LLM endpoint：

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_sft
export GSI_DISABLE_TOKEN_STATS=1
```

TANGO 后端：

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

数据集与离线缓存：

```bash
export GSI_DATASET_ROOT=/GSI/dataset
export HF_HUB_OFFLINE=1
```

## 常见配方

训练后模型评估：

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_sft
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120

python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 4 \
  --max-count 100 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/eval_multi_method/qwen3_0_6b_cybertown
```

采集 replan 数据：

```bash
python run/run_collect_replan_dataset.py \
  --batch-name qwen3_0_6b_cybertown_nc4 \
  --output-root outputs/replan_collect \
  --dataset-root "$GSI_DATASET_ROOT" \
  --max-count 100 \
  --max-workers 10 \
  --max-newcases-per-run 4 \
  --capture-save-llm-io
```

## 排查要点

- `GSI_LLM_MODEL` 必须匹配 vLLM 暴露名。
- `model_name_override` 非 `null` 时会覆盖环境变量模型名。
- `solver_type` 选择 planner；`GSI_TANGO_SOLVER_BACKEND` 选择 allocator 求解器。
- `enable_new_case_generation=false` 时，newcase 相关参数不会影响普通执行。
- `collect_replan_dataset=true` 会改变输出和采集行为。
- `platform_type=unreal` 要求 UE 服务可访问。
- `--max-workers` 过高会导致模型服务超时、输出截断或失败率异常。
