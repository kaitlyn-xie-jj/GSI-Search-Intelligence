# Installation and Environment

GSI can run in a local Python environment or in the Docker training image. Evaluation and training require a model endpoint, a task dataset, a TANGO solver backend, and a Hugging Face cache.

## Python Environment

Python 3.10 is recommended:

```bash
conda create -n gsi python=3.10
conda activate gsi
pip install -r requirements.txt
```

The documentation site also requires:

```bash
pip install mkdocs-material
```

## Docker Environment

Training and evaluation are recommended through the repository Docker setup. Inside the container, the common working directory is:

```text
/GSI
```

Check the runtime environment:

```bash
cd /GSI
check_env.sh
```

Docker image, cache mount, and proxy details are documented in `docker/README.runtime-train.md`.

## LLM Endpoint

GSI calls model services through an OpenAI-compatible API. Common variables:

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
```

`GSI_LLM_MODEL` must match the model id returned by `/v1/models`.

## Solver Backend

The TANGO allocator uses SCIP by default:

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

If a valid Gurobi license is available:

```bash
export GSI_TANGO_SOLVER_BACKEND=gurobi
```

Troubleshooting is covered in [Solver](../troubleshooting/solver.md).

## Dataset

The benchmark dataset is:

```text
WindyLab/GSI
```

Explicitly set the local dataset path before running scripts:

```bash
export GSI_DATASET_ROOT=/path/to/gsi/dataset
```

Hugging Face cache and offline mode are covered in [Use Hugging Face Models and Data](../tutorials/use-hf-models-datasets.md).
