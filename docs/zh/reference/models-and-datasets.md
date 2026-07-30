# 模型与数据集

本页列出文档中使用的公开模型和数据集。实际运行时建议固定 repo id、revision 或本地 snapshot 路径。

## 模型

| 用途 | Repo |
| --- | --- |
| 基础模型 | `Qwen/Qwen3-0.6B` |
| SFT 后模型 | `WindyLab/Qwen3-0.6B-cybertown-SFT` |
| RLVR 后模型 | `WindyLab/Qwen3-0.6B-cybertown-RLVR` |

## 数据集

| 用途 | Repo |
| --- | --- |
| Benchmark / replan 采集 | `WindyLab/GSI` |
| SFT 训练 | `WindyLab/Qwen3-0.6B-cybertown-SFT-data` |
| RLVR 训练 | `WindyLab/Qwen3-0.6B-cybertown-RLVR-data` |

## 使用建议

- Benchmark 使用 `WindyLab/GSI`，应与 SFT/RLVR 训练数据区分管理。
- 复现实验时记录 snapshot 路径或 revision。
- 离线运行前确认容器可见的 Hugging Face cache 已包含所需文件。

下载方式见 [Hugging Face 准备](../training/huggingface-prepare.md)。
