# 运行 Benchmark

批量 benchmark 使用同一批任务运行一个或多个 planner 方法。入口为：

```bash
python run/run_exp_multi_method.py
```

## 基础 SGI 运行

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 100 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/batch_runs
```

## 多方法对比

```bash
python run/run_exp_multi_method.py \
  --methods sgi spine smartllm lipllm \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/batch_runs
```

支持的方法包括：

```text
sgi
spine
smartllm
lipllm
```

## 任务筛选

常用参数：

```bash
--task-mix
--plan-level
--coor-level
--lang-level
--max-count
```

示例：

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --task-mix transport=0.5 assembly=0.5 \
  --max-count 20 \
  --dataset-root "$GSI_DATASET_ROOT"
```

## New Case

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 1 2 3 4 \
  --dataset-root "$GSI_DATASET_ROOT"
```

## 输出结构

普通 benchmark：

```text
results/batch_runs/<timestamp>/
  batch_sgi/
  batch_spine/
```

New case benchmark：

```text
results/batch_runs/<timestamp>/
  newcase_0/
  newcase_1/
```

输出文件说明见 [输出目录](../training/outputs.md)。
