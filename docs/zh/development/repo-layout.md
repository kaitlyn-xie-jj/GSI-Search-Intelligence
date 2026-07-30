# 仓库结构

GSI 仓库可按运行链路划分为配置、数据、核心模块、运行入口和训练脚本。

```text
config/        默认运行配置
docs/          文档网站内容
modules/       planner、validator、solver、world model 和 platform
run/           benchmark、评估和数据采集入口
llm_finetune/  SFT、RLVR、vLLM、VeRL 和训练容器脚本
```

## 核心模块

```text
modules/task_solver/
```

任务求解主路径，包含 SGI planner、baseline planner、solver factory 和 world model。

```text
modules/plan_validator/
```

计划验证、replan state 加载和 RLVR reward 相关逻辑。

```text
modules/platform/
```

Semantic platform 与 Unreal platform 的执行抽象。

```text
modules/dataset_builder/
modules/dataset_loader/
```

数据构建、加载和筛选逻辑。

## 运行入口

```text
run/run_exp_multi_method.py
run/run_collect_replan_dataset.py
run/plan_validation_server.py
```

## 训练入口

```text
llm_finetune/scripts/runtime/train_sft_unsloth.sh
llm_finetune/scripts/runtime/train_rlvr.sh
llm_finetune/scripts/runtime/serve_validator.sh
```

训练流程见 [训练与模型概览](../training/overview.md)。

## 修改建议

- 新增 planner 时，优先接入 `modules/task_solver/` 的统一生命周期。
- 新增 validator 规则时，同时检查 benchmark 和 RLVR reward 的调用路径。
- 新增任务数据时，确保 goal、scenario、success condition 和筛选逻辑一致。
- 修改公开 CLI 参数、数据格式或模型仓库时，同步更新文档。
