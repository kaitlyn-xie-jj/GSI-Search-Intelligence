# Ray 与 Checkpoint

RLVR 训练依赖 Ray、VeRL 和 vLLM。异常中断后，旧进程和旧 checkpoint 是最常见的问题来源。

## 清理旧 Ray

训练前可启用 preflight：

```bash
GSI_PREFLIGHT=1
GSI_PREFLIGHT_STOP_RAY=1
```

`train_rlvr.sh` 会在 preflight 中尝试清理旧 Ray runtime。

## 自动恢复 Checkpoint

VeRL 默认 `trainer.resume_mode=auto`。如果 `RLVR_OUTPUT_DIR` 中已有 `global_step_*`，重新启动会自动恢复。

日志示例：

```text
Found checkpoint: .../global_step_1800
Load from checkpoint folder: .../global_step_1800
Setting global step to 1800
Resuming from .../global_step_1800
```

## 新实验使用新目录

修改 batch、rollout、epoch 或数据集后，建议使用新的输出目录：

```bash
export RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr_bsz8
```

强制忽略旧 checkpoint：

```bash
trainer.resume_mode=disable
```

完整说明见 [RLVR 训练](../training/rlvr.md)。
