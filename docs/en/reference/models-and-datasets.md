# Models and Datasets

This page lists the public models and datasets used in the documentation. Formal runs should fix the repo id, revision, or local snapshot path.

## Models

| Use | Repo |
| --- | --- |
| Base model | `Qwen/Qwen3-0.6B` |
| SFT model | `WindyLab/Qwen3-0.6B-cybertown-SFT` |
| RLVR model | `WindyLab/Qwen3-0.6B-cybertown-RLVR` |

## Datasets

| Use | Repo |
| --- | --- |
| Benchmark / replan collection | `WindyLab/GSI` |
| SFT training | `WindyLab/Qwen3-0.6B-cybertown-SFT-data` |
| RLVR training | `WindyLab/Qwen3-0.6B-cybertown-RLVR-data` |

## Usage Guidance

- Benchmark uses `WindyLab/GSI`; manage it separately from SFT/RLVR training data.
- Record the snapshot path or revision for reproduction.
- Before offline runs, confirm that the container-visible Hugging Face cache contains all required files.

Download instructions are in [Hugging Face Preparation](../training/huggingface-prepare.md).
