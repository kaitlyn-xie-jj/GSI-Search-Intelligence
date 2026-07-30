# 复现实验结果

GSI 实验结果受代码版本、数据集、模型 checkpoint、vLLM 模型名、任务筛选、newcase 设置、solver 后端和并发配置影响。正式复现实验前，应固定这些变量并记录完整命令。

## 复现路线

1. Benchmark 复现：使用已发布模型评估 SGI。
2. 多方法对比：在同一批任务上比较 SGI 与 baseline planner。
3. New Case / Replan 复现：评估突发事件和重规划能力。
4. 训练复现：重新运行 SFT 或 RLVR，再评估得到的 checkpoint。

前三类属于评估流程；训练复现属于模型训练流程。两类输出目录应分开管理。

## 需要固定的变量

| 变量 | 建议记录方式 | 说明 |
| --- | --- | --- |
| 代码版本 | `git rev-parse HEAD` | 运行逻辑、prompt 和指标聚合可能随代码变化。 |
| 数据集 | `GSI_DATASET_ROOT` 或 snapshot 路径 | 保证任务集合和筛选结果一致。 |
| 模型 | repo id 或 checkpoint 路径 | 区分 base、SFT 和 RLVR 模型。 |
| 模型服务名 | `/v1/models` 返回的 id | 必须等于 `GSI_LLM_MODEL`。 |
| 命令 | 完整 shell 命令 | 包含方法、样本数、筛选条件和 newcase 设置。 |
| solver 后端 | `GSI_TANGO_SOLVER_BACKEND` | 影响任务分配结果和超时行为。 |
| 并发 | `--max-workers`、`--max-num-seqs` | 影响模型服务稳定性。 |
| 输出目录 | `--output-root` 和 timestamp | 后续对比必须指向同一批输出。 |

## 1. 评估已发布 RLVR 模型

设置模型：

```bash
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
export MODEL_PATH=WindyLab/Qwen3-0.6B-cybertown-RLVR
```

启动 vLLM 后检查：

```bash
curl -fsS http://127.0.0.1:8001/v1/models | python -m json.tool
```

配置 GSI：

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

准备数据集：

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

先运行 smoke：

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --task-mix transport=1.0 \
  --max-count 5 \
  --max-workers 1 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/reproduce/smoke_rlvr
```

再运行正式 benchmark：

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/reproduce/sgi_rlvr
```

并发过高会增加请求超时和格式错误风险。单个本地 vLLM endpoint 上应先确认稳定性，再提高 `--max-workers`。

## 2. 多方法对比

`run_exp_multi_method.py` 会先确定任务集合，再对各方法运行同一批任务：

```bash
python run/run_exp_multi_method.py \
  --methods sgi spine smartllm lipllm \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/reproduce/multi_method_rlvr
```

输出结构：

```text
results/reproduce/multi_method_rlvr/<timestamp>/
  batch_sgi/
  batch_spine/
  batch_smartllm/
  batch_lipllm/
```

对比时至少记录：

- `success_rate`
- `elapsed_sec`
- `llm_calls`
- `replans_total`
- `total_energy`
- `prompt_tokens_total` / `response_tokens_total`

指标定义见 [评估指标](../concepts/evaluation-metrics.md)。

## 3. New Case / Replan 评估

普通 benchmark 不注入 new case。开启方式：

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 1 2 3 4 \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/reproduce/newcase_rlvr
```

输出中 `newcase_0` 是无主动注入的基线：

```text
results/reproduce/newcase_rlvr/<timestamp>/
  newcase_0/batch_sgi/
  newcase_1/batch_sgi/
  newcase_2/batch_sgi/
  newcase_3/batch_sgi/
  newcase_4/batch_sgi/
```

重点比较：

- `success_rate`
- `replans_full` / `replans_partial`
- `newcase_total`
- `newcase_top_types`
- `elapsed_sec`
- `llm_calls`

## 4. 训练复现

训练复现分为两步：

1. SFT：从 base model 学习任务规划格式、约束和输出结构。
2. RLVR：从 SFT 后模型出发，用 validator/reward 信号继续优化。

入口：

- [SFT 训练](../training/sft.md)
- [RLVR 训练](../training/rlvr.md)
- [Hugging Face 准备](../training/huggingface-prepare.md)

训练日志不是 benchmark 结果。训练完成后，应通过 vLLM 启动模型，再按本页评估流程运行。

## 查看输出

每个 `batch_<method>/` 下主要文件为：

- `summary.jsonl`：每个任务的一条结果，适合定位失败样本。
- `aggregate_full.json`：聚合指标，适合生成表格和方法对比。
- `0001_<task_id>/`：单任务运行目录，包含日志、临时变量、LLM 输出和执行中间状态。

快速查看最新结果：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("results/reproduce/sgi_rlvr")
latest = max(root.iterdir(), key=lambda p: p.stat().st_mtime)
agg = latest / "batch_sgi" / "aggregate_full.json"
data = json.loads(agg.read_text())
print(agg)
print(json.dumps(data["overall"], ensure_ascii=False, indent=2))
PY
```

完整说明见 [输出目录](../training/outputs.md)。

## 结果不一致时的检查顺序

1. `/v1/models` 返回的模型名是否等于 `GSI_LLM_MODEL`。
2. 是否混用了 base、SFT 和 RLVR 模型。
3. `GSI_DATASET_ROOT` 是否指向同一个 snapshot。
4. `--max-count`、`--task-mix`、`--plan-level`、`--coor-level`、`--lang-level` 是否一致。
5. 是否开启了 `--enable-newcase`。
6. `GSI_TANGO_SOLVER_BACKEND` 和 `GSI_TANGO_SOLVER_MAX_TIME` 是否一致。
7. `--max-workers` 是否使 vLLM 过载。
8. 输出目录是否混入不同实验。
