# 新增 Validator 规则

Validator 用于检查计划格式、技能合法性、状态约束和 reward 计算。新增规则时，应保证 benchmark、validator server 和 RLVR reward 的判断一致。

## 关键路径

```text
modules/plan_validator/
run/plan_validation_server.py
llm_finetune/rlvr/gsi_reward_manager.py
```

## 接入要求

- 输入来自 task、state、plan 或 execution feedback。
- 输出包含明确错误原因，便于 replan 和 reward 分析。
- 训练 reward 与离线 validator 使用相同语义。
- 依赖 state shard 的规则需要同步更新 state 构建流程。

## 验证

启动 validator：

```bash
python run/plan_validation_server.py
```

训练容器中检查服务：

```bash
./scripts/runtime/serve_validator.sh status
```

训练侧说明见 [RLVR 训练](../training/rlvr.md)。
