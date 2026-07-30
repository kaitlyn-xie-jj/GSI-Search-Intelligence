# Add a Validator Rule

The validator checks plan format, skill legality, state constraints, and reward computation. New rules should behave consistently across benchmark, validator server, and RLVR reward.

## Key Paths

```text
modules/plan_validator/
run/plan_validation_server.py
llm_finetune/rlvr/gsi_reward_manager.py
```

## Integration Requirements

- Inputs should come from task, state, plan, or execution feedback.
- Outputs should include explicit error reasons for replan and reward analysis.
- Training reward and offline validation should use the same semantics.
- Rules that depend on state shards must update the state build flow.

## Validation

Start the validator:

```bash
python run/plan_validation_server.py
```

Check the service in the training container:

```bash
./scripts/runtime/serve_validator.sh status
```

Training-side details are in [RLVR Training](../training/rlvr.md).
