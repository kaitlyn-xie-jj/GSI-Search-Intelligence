# 训练与模型概览

GSI 的模型流程分为资源准备、SFT、RLVR 和评估四个阶段。

## 流程

1. 准备 Hugging Face 模型、数据集和本地 cache。
2. 使用 SFT 训练模型遵循任务规划格式和输出约束。
3. 使用 RLVR 在 validator/reward 信号下继续优化。
4. 通过 vLLM 启动模型，并用 GSI benchmark 评估。

## 推荐阅读顺序

1. [Hugging Face 准备](huggingface-prepare.md)
2. [SFT 训练](sft.md)
3. [RLVR 训练](rlvr.md)
4. [vLLM 启动与 GSI 评估](vllm-eval.md)
5. [输出目录](outputs.md)

## 范围说明

本节保留训练、checkpoint、validator、vLLM、GPU 和输出分析的操作细节。系统概念、配置字段和开发扩展分别见 Concepts、Reference 和 Development。
