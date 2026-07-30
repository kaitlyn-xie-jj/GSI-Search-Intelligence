# 安装与环境

GSI 可在本地 Python 环境或 Docker 训练镜像中运行。评估和训练依赖模型服务、任务数据集、TANGO solver 后端和 Hugging Face cache。

## Python 环境

推荐 Python 3.10：

```bash
conda create -n gsi python=3.10
conda activate gsi
pip install -r requirements.txt
```

构建文档网站还需要：

```bash
pip install mkdocs-material
```

## Docker 环境

训练和评估推荐使用仓库提供的 Docker 配置。进入容器后，常用工作目录为：

```text
/GSI
```

运行前检查环境：

```bash
cd /GSI
check_env.sh
```

Docker 镜像、cache 挂载和代理配置见仓库中的 `docker/README.runtime-train.md`。

## LLM Endpoint

GSI 通过 OpenAI-compatible API 调用模型服务。常用环境变量：

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
```

`GSI_LLM_MODEL` 必须与 `/v1/models` 返回的模型名一致。

## Solver 后端

TANGO allocator 默认推荐使用 SCIP：

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

如果机器具备有效 Gurobi license，可切换为：

```bash
export GSI_TANGO_SOLVER_BACKEND=gurobi
```

排查方式见 [Solver](../troubleshooting/solver.md)。

## 数据集

Benchmark 数据集为：

```text
WindyLab/GSI
```

建议显式解析数据集路径，并在运行脚本时传入：

```bash
export GSI_DATASET_ROOT=/path/to/gsi/dataset
```

Hugging Face cache 和离线模式见 [使用 Hugging Face 模型和数据](../tutorials/use-hf-models-datasets.md)。
