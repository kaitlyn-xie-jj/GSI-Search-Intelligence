# CLI and Scripts

This page lists common entry points. Use `--help` for the full argument list.

## Benchmark

```bash
python run/run_exp_multi_method.py --help
```

Use cases:

- Run SGI or baseline planners.
- Set task type, difficulty filters, and sample count.
- Enable new-case evaluation.
- Specify the output directory.

## Replan Data Collection

```bash
python run/run_collect_replan_dataset.py --help
```

Use cases:

- Run tasks in batch.
- Inject new cases.
- Capture replan prompt, response, and state.
- Generate sources for later training data.

## Validator Server

```bash
python run/plan_validation_server.py
```

In the training container, use the wrapper:

```bash
./scripts/runtime/serve_validator.sh start
./scripts/runtime/serve_validator.sh status
./scripts/runtime/serve_validator.sh logs
./scripts/runtime/serve_validator.sh stop
```

## SFT / RLVR

```bash
./scripts/runtime/train_sft_unsloth.sh
./scripts/runtime/train_rlvr.sh
```

See [SFT Training](../training/sft.md) and [RLVR Training](../training/rlvr.md).
