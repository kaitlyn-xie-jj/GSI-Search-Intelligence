# 输出目录

本页说明 `run/run_exp_multi_method.py` 产生的评估输出。典型路径：

```text
outputs/eval_multi_method/qwen3_0_6b_cybertown/<timestamp>/newcase_0/batch_sgi/
```

如果使用 Docker compose，宿主机上可能对应：

```text
outputs/docker_runtime_train/eval_multi_method/qwen3_0_6b_cybertown/<timestamp>/newcase_0/batch_sgi/
```

## 目录结构

```text
outputs/eval_multi_method/<eval_name>/
  <timestamp>/
    newcase_0/
      batch_sgi/
        aggregate_full.json
        summary.jsonl
        0001_<task_id>/
          metrics.json
          single_run_summary.json
          temp_vars.jsonl
          log.md
          alloc_config/
            planner_param.yaml
            scene.yaml
            task_param.yaml
            vehicle_param.yaml
          alloc_results.yaml
    newcase_4/
      batch_sgi/
        aggregate_full.json
        summary.jsonl
```

层级含义：

- `<eval_name>`：`--output-root` 下的实验目录。
- `<timestamp>`：一次运行的时间戳。
- `newcase_N`：最多主动注入 N 个 new case；普通 benchmark 可能没有该层。
- `batch_<method>`：某个 planner 方法的批量结果。
- `0001_<task_id>`：单个任务的运行目录。

## 关键文件

### `aggregate_full.json`

批次聚合指标。用于论文表格、方法对比和 newcase 强度对比。

常见字段：

- `overall`：整体指标。
- `by_goal_type`：按任务类型聚合。
- `input_params`：运行参数快照。

### `summary.jsonl`

每行对应一个 task run。适合筛选失败样本、定位 task id 和 run 目录。

常见字段：

- `goal_type`
- `goal_id`
- `run_dir`
- `metrics`
- `algorithm_specific`

### `single_run_summary.json`

单任务摘要，通常包含运行参数、目标信息、成功状态、耗时和关键 metrics。

### `metrics.json`

单任务指标文件。适合查看成功状态、耗时、replan 次数、能耗和 token 统计。

### `log.md`

面向人工阅读的运行日志。适合查看 planner 输出、执行步骤、反馈和失败原因。

### `temp_vars.jsonl`

调试变量快照，每行一个 JSON 对象。常见记录：

- `run_input_default`
- `goal`
- `prompt`
- `response`
- `dispatcher_result`
- `skills_by_timestep`
- `execution_result`
- `feedback_event`
- `new_case_event`

当 `log.md` 过长或需要脚本筛选变量时，优先使用该文件。

### `alloc_config/`

传给 TANGO allocator 的配置快照：

- `planner_param.yaml`
- `task_param.yaml`
- `vehicle_param.yaml`
- `scene.yaml`

用于复现或排查任务分配问题。

### `alloc_results.yaml`

TANGO allocator 的输出。常见字段：

- `result.flagSuccess`
- `energyConstraintCheck`
- `energyCost`
- `finalAllCost`
- `optStatus`
- `objVal`
- `MIPGap`
- `solverTime`
- `team`
- `vehicle_paths`

如果该文件不存在，通常表示 run 未进入 allocation 阶段，或在 allocation 前失败。

## 聚合指标

`aggregate_full.json` 中常见指标：

| 字段 | 含义 |
| --- | --- |
| `count_runs` | 统计的 run 数。 |
| `success_rate` | 成功 run 的比例。 |
| `elapsed_sec` | 单个 run 总耗时。 |
| `llm_calls` | LLM 调用次数。 |
| `planning_duration` | 规划耗时。 |
| `allocation_duration` | 分配耗时。 |
| `replans_full` | 完整重规划次数。 |
| `replans_partial` | 局部重规划次数。 |
| `replans_total` | 重规划总次数。 |
| `total_energy` | 累计能耗或移动代价。 |
| `newcase_total_orig` | 主动注入的 new case 数。 |
| `newcase_total` | 实际记录的 new case 数。 |
| `newcase_top_types` | 高频 new case 类型。 |
| `completed_tasks` | 已完成 atomic tasks 数。 |
| `total_tasks` | atomic tasks 总数。 |
| `atomic_task_completion_rate` | atomic task 完成比例。 |
| `prompt_tokens_total` | prompt token 总数。 |
| `response_tokens_total` | response token 总数。 |

Token 统计依赖模型服务支持；设置 `GSI_DISABLE_TOKEN_STATS=1` 时通常不可用。

## 比较结果

比较 newcase 强度：

```text
newcase_0/batch_sgi/aggregate_full.json
newcase_4/batch_sgi/aggregate_full.json
```

重点看：

- `success_rate`
- `replans_total`
- `elapsed_sec`
- `llm_calls`
- `atomic_task_completion_rate`
- `by_goal_type`

比较不同方法：

```text
newcase_0/batch_sgi/
newcase_0/batch_spine/
newcase_0/batch_smartllm/
newcase_0/batch_lipllm/
```

评估 SFT/RLVR 模型时通常先看 `batch_sgi`；baseline 方法更适合基础模型对比。

## 快速查看

查看聚合文件：

```bash
RUN=outputs/eval_multi_method/qwen3_0_6b_cybertown/2026-06-04_19-02-36

python -m json.tool "$RUN/newcase_0/batch_sgi/aggregate_full.json" | less
python -m json.tool "$RUN/newcase_4/batch_sgi/aggregate_full.json" | less
```

查看第一条 summary：

```bash
head -1 "$RUN/newcase_0/batch_sgi/summary.jsonl" | python -m json.tool
```

列出失败任务：

```bash
python - <<'PY'
import json
from pathlib import Path

summary = Path("outputs/eval_multi_method/qwen3_0_6b_cybertown/2026-06-04_19-02-36/newcase_0/batch_sgi/summary.jsonl")
for line in summary.read_text().splitlines():
    row = json.loads(line)
    if not row.get("metrics", {}).get("success", False):
        print(row["goal_type"], row["goal_id"], row["run_dir"])
PY
```

按任务类型汇总：

```bash
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

summary = Path("outputs/eval_multi_method/qwen3_0_6b_cybertown/2026-06-04_19-02-36/newcase_0/batch_sgi/summary.jsonl")
stats = defaultdict(lambda: [0, 0])
for line in summary.read_text().splitlines():
    row = json.loads(line)
    key = row.get("goal_type", "unknown")
    stats[key][1] += 1
    stats[key][0] += int(bool(row.get("metrics", {}).get("success", False)))
for key, (ok, total) in sorted(stats.items()):
    print(f"{key}: {ok}/{total} = {ok / total:.2%}")
PY
```

查看失败 run：

```bash
CASE=outputs/eval_multi_method/qwen3_0_6b_cybertown/2026-06-04_19-02-36/newcase_0/batch_sgi/0001_cybertown_scenario_1_g_19075
python -m json.tool "$CASE/single_run_summary.json" | less
less "$CASE/log.md"
```

## 常见误读

- `ok=true` 不等于任务成功；任务成功看 `metrics.success` 或 `success_rate`。
- `summary.jsonl` 行顺序可能受并发影响，不一定等于 task 编号顺序。
- `newcase_0` 下 `newcase_total_orig` 应为 0；`newcase_total` 非 0 表示执行中记录到事件。
- `aggregate_full.json` 只有批次结束后才完整写出。
- `alloc_config/` 或 `alloc_results.yaml` 缺失通常表示未进入 allocation 阶段。
- Token 统计关闭时，相关字段为空或不可靠。
