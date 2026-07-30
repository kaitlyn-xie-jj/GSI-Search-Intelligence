# GSI LLM 微调入口

这份 README 是 `llm_finetune/` 的容器内部入口索引。它假设你已经进入训练容器，仓库挂载在 `/GSI`，并且后续命令都在容器内执行。

如果你还在宿主机上，需要先看仓库根目录下的 `docker/README.runtime-train.md` 完成镜像构建、Hugging Face cache 挂载和容器启动。进入容器后再回到这里。

## 容器内部初始化

进入容器后先确认当前路径和基础环境：

```bash
cd /GSI/llm_finetune
export ROOT_DIR=/GSI
./scripts/runtime/check_env.sh
```

`check_env.sh` 会检查 PyTorch、Transformers、Unsloth、VeRL、vLLM、Ray、Gurobi 等关键依赖能否 import。这个检查通过，只能说明环境能启动；真正的 SFT/RLVR 还需要模型、数据、Hugging Face cache 和输出目录挂载正确。

训练容器默认保留网络和代理环境，可以访问 Hugging Face。只有在需要纯本地缓存运行时，才显式设置 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`WANDB_MODE=offline`。

## 文档入口

训练相关文档统一放在仓库根目录的 `../docs/training/`。目录入口是 `../docs/training/README.md`。

- `../docs/training/huggingface_prepare.md`：训练和评估前的 Hugging Face cache 准备，包含基础模型、SFT/RLVR checkpoint、训练数据集和 GSI benchmark 数据集。
- `../docs/training/sft_training.md`：SFT 训练，包含 Unsloth 常规训练、换 GPU、本地路径训练和 LoRA 合并。
- `../docs/training/rlvr_training.md`：RLVR 训练，包含 RLVR 数据要求、validator、preflight、VeRL 参数、smoke test 和输出目录。
- `../docs/training/vllm_eval.md`：训练后用 vLLM 启动完整模型、SFT LoRA、RLVR LoRA，并连接 GSI 评估脚本。
- `../docs/training/eval_outputs.md`：`run_exp_multi_method.py` 输出目录、文件和统计字段说明。

推荐顺序：

1. 先看 `../docs/training/huggingface_prepare.md`，确认缓存、权限和离线模式。
2. 跑 SFT 时看 `../docs/training/sft_training.md`。
3. 跑 RLVR 时看 `../docs/training/rlvr_training.md`。
4. 训练后评估看 `../docs/training/vllm_eval.md`。
5. 评估结束后看 `../docs/training/eval_outputs.md`。
