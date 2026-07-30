# GSI 统一运行/训练镜像

这是 GSI 本地和远程工作流共用的一套 Docker 镜像。这个镜像用于：

- GSI 运行和 benchmark 脚本
- vLLM 模型服务
- plan validator / RLVR reward validation
- SFT 训练
- RLVR 训练
- LoRA 合并和训练后工具命令

镜像里只放运行和训练依赖，不把 GSI 仓库、模型权重、SFT/RLVR 数据、任务数据集或输出结果打进镜像。上述内容都通过运行时挂载进入容器，所以宿主机上的代码和数据变化后，不需要重新 build 镜像。

## 准备 Hugging Face 缓存

建议使用一个持久化的宿主机缓存目录，并挂载到容器内的 `/root/.cache/huggingface`。如果容器运行时有网络，提前下载不是必须的；但提前下载可以减少重复启动 SFT/RLVR 和 benchmark / replan 采集 / multi-method 评估的等待时间，也方便做只读本地缓存运行。

```bash
cd /path/to/GSI

python -m pip install -U huggingface_hub

export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"

# quickstart / benchmark 必需资源。
hf download --repo-type dataset --revision small WindyLab/GSI
hf download WindyLab/Qwen3-0.6B-cybertown-RLVR
```

如果要训练，再下载训练相关资源：

```bash
# SFT / RLVR 训练数据。
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-SFT-data
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-RLVR-data

# RLVR 训练常用 SFT 起点；评估 SFT 模型时也会用到。
hf download WindyLab/Qwen3-0.6B-cybertown-SFT

# 如果要从 base model 重新跑 fresh SFT，再下载这个基础模型。
hf download Qwen/Qwen3-0.6B
```

`WindyLab/GSI` 是运行 benchmark、replan 采集和 multi-method 评估时使用的任务数据集，不是 SFT/RLVR 训练数据。`WindyLab/Qwen3-0.6B-cybertown-RLVR` 是可直接用 vLLM 评估的 RLVR 后完整模型。

已发布的仓库如下：

```text
Model:   WindyLab/Qwen3-0.6B-cybertown-SFT
Model:   WindyLab/Qwen3-0.6B-cybertown-RLVR
Dataset: WindyLab/Qwen3-0.6B-cybertown-SFT-data
Dataset: WindyLab/Qwen3-0.6B-cybertown-RLVR-data
Dataset: WindyLab/GSI (revision: small)
```

RLVR 数据集使用分片 state record：`train.parquet`、`val.parquet`、`manifest.json`、`states.index.json`、`states.shards.manifest.json` 和 `states/*.jsonl`。

进入容器后，如果不把任务数据集显式挂载到 `/GSI/dataset`，评估文档会用 Hugging Face cache 解析 `WindyLab/GSI` 的 snapshot 路径，并通过 `--dataset-root` 传给评估脚本。

如果公开仓库下载时返回 `401 Unauthorized`，先检查当前 `HF_HOME` 下的登录状态。切换过 `HF_HOME` 后可能带着旧 token 或没有 token；可以在同一个 shell 里运行 `hf auth logout` 后再 `hf auth login`，然后重试下载。

## 构建和进入容器

以下命令在宿主机运行。先构建统一镜像：

```bash
cd /path/to/GSI
docker compose -f docker/docker-compose.runtime-train.yml build gsi-train
```

如果模型、数据或输出目录不在仓库父目录下，先在宿主机设置挂载路径。这些变量只在创建容器时生效：

```bash
export GSI_HOST_ROOT=${GSI_HOST_ROOT:-$PWD}
export GSI_REPO_ROOT=$GSI_HOST_ROOT
export GSI_MODEL_ROOT=$GSI_HOST_ROOT/../models
export GSI_DATA_ROOT=$GSI_HOST_ROOT/data
export GSI_DATASET_ROOT=$GSI_HOST_ROOT/dataset
export GSI_OUTPUT_ROOT=$GSI_HOST_ROOT/outputs/docker_runtime_train
export HF_CACHE_ROOT=${HF_HOME:-$GSI_HOST_ROOT/hf_cache}
```

进入训练容器：

```bash
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
```

如果之前改过 compose 服务名，可能会残留旧容器。回到宿主机运行命令时可以顺手清理：

```bash
docker compose -f docker/docker-compose.runtime-train.yml run --rm --remove-orphans gsi-train bash
```

## 容器内初始化和检查

进入容器后，后续训练、validator、vLLM 和评估命令都在容器内执行；不要在容器里再次运行 `docker compose`。

容器内默认工作目录是 `/GSI`。检查容器是否能导入运行、SFT、RLVR、vLLM、Ray 和数据处理依赖：

```bash
check_env.sh
```

期望最后一行是：

```text
GSI runtime/training environment OK
```

compose 服务会把宿主机仓库挂载到容器内 `/GSI`，并使用挂载进来的 `llm_finetune/scripts/runtime/entrypoint.sh` 作为入口脚本。entrypoint 只负责初始化环境并执行你传入的真实命令；训练、验证、vLLM 和评估命令放在容器内文档 `/GSI/llm_finetune/README.md`。

主要挂载关系：

```text
${GSI_REPO_ROOT:-..}                         -> /GSI
${GSI_MODEL_ROOT:-../models}                 -> /models:ro
${GSI_DATA_ROOT:-../data}                    -> /GSI/data:ro
${GSI_DATASET_ROOT:-../dataset}              -> /GSI/dataset:ro
${GSI_OUTPUT_ROOT:-../outputs/docker_runtime_train} -> /GSI/outputs
${HF_CACHE_ROOT:-../hf_cache}                -> /root/.cache/huggingface
```

`docker/docker-compose.runtime-train.yml` 位于仓库的 `docker/` 目录下，所以默认的 `../hf_cache` 对应从仓库根目录看到的 `$PWD/hf_cache`。这和上面 `HF_HOME=$PWD/hf_cache` 的下载位置一致；通常不需要额外设置 `HF_CACHE_ROOT`。

compose 服务使用 host network，并会把宿主机当前的 `HTTP_PROXY`、`HTTPS_PROXY`、`http_proxy`、`https_proxy` 透传进容器，方便访问 Hugging Face。host network 下，容器里的 `127.0.0.1:7897` 会指向宿主机本机代理。`NO_PROXY/no_proxy` 只保留 `127.0.0.1,localhost`，用于 validator 等本地服务。

如果宿主机代理监听 `127.0.0.1:7897`，可以这样启动：

```bash
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
```

如果需要只使用本地缓存运行，在容器内显式设置这些环境变量：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export PIP_NO_INDEX=1
check_env.sh
```

进入容器后切到训练入口目录：

```bash
cd /GSI/llm_finetune
```

后续 SFT、RLVR、vLLM 和评估命令看容器内文档：`/GSI/llm_finetune/README.md`。
