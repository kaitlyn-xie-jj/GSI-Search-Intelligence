# Replan 与 New Case

Replan 和 new case 用于评估系统在动态任务条件下的恢复能力。普通 benchmark 可以不注入 new case；鲁棒性评估需要显式开启。

## New Case

New case 指执行过程中出现的新情况，例如目标变化、机器人故障、道路阻塞或环境变化。

相关路径：

```text
modules/platform/semantic_platform/new_case_generator.py
modules/platform/semantic_platform/new_case_injector.py
modules/platform/semantic_platform/new_case_controller.py
```

Benchmark 开启方式：

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 1 2 3 4
```

## Replan

Replan 指系统根据执行反馈重新生成或调整计划。常见触发条件包括：

- 主目标未完成。
- 技能执行失败。
- new case 改变任务条件。
- validator 或 goal monitor 判定当前计划不可继续。

配置示例：

```json
{
  "enable_replanning": true,
  "enable_new_case_generation": false
}
```

## 数据采集

采集 replan 数据：

```bash
python run/run_collect_replan_dataset.py --help
```

采集结果可整理为 SFT/RLVR 数据。训练侧说明见 [Hugging Face 准备](../training/huggingface-prepare.md) 和 [RLVR 训练](../training/rlvr.md)。
