# Training Overview

The GSI model workflow has four stages: resource preparation, SFT, RLVR, and evaluation.

## Flow

1. Prepare Hugging Face models, datasets, and local cache.
2. Use SFT to teach the model task planning format and output constraints.
3. Use RLVR to continue optimization with validator/reward signals.
4. Start the model with vLLM and evaluate it through GSI benchmark.

## Recommended Reading Order

1. [Hugging Face Preparation](huggingface-prepare.md)
2. [SFT Training](sft.md)
3. [RLVR Training](rlvr.md)
4. [vLLM Startup and GSI Evaluation](vllm-eval.md)
5. [Output Directory](outputs.md)

## Scope

This section keeps operational details for training, checkpoints, validator, vLLM, GPU, and output analysis. System concepts, configuration fields, and development extensions are covered in Concepts, Reference, and Development.
