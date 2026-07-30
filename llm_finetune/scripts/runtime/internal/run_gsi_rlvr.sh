#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

unset RAY_ADDRESS
if [ "${VERL_RAY_STOP_BEFORE_START:-1}" = "1" ]; then
  "${PYTHON_BIN}" -m ray.scripts.scripts stop || true
fi

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

REWARD_FILE_PATH="${REWARD_FILE_PATH:-${ROOT_DIR}/llm_finetune/rlvr/gsi_reward_manager.py}"
DATA_DIR="${RLVR_DATA_DIR:-${ROOT_DIR}/data/rlvr_gsi/verl_newcase_prompt}"
MODEL_PATH="${RLVR_MODEL_PATH:-${ROOT_DIR}/models/Qwen3/Qwen3-0.6B}"
OUTPUT_DIR="${RLVR_OUTPUT_DIR:-${ROOT_DIR}/outputs/rlvr_gsi_dr_grpo}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
  echo "RLVR train parquet does not exist: ${DATA_DIR}/train.parquet" >&2
  exit 1
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  trainer.val_before_train=False \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.train_batch_size="${VERL_TRAIN_BATCH_SIZE:-64}" \
  data.max_prompt_length="${VERL_MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${VERL_MAX_RESPONSE_LENGTH:-1680}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  data.filter_overlong_prompts_workers=32 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank="${VERL_LORA_RANK:-64}" \
  actor_rollout_ref.model.lora_alpha="${VERL_LORA_ALPHA:-32}" \
  actor_rollout_ref.actor.optim.lr="${VERL_LR:-1e-5}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${VERL_PPO_MINI_BATCH_SIZE:-8}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${VERL_PPO_MICRO_BATCH_SIZE_PER_GPU:-4}" \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
  actor_rollout_ref.actor.loss_scale_factor=2048 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${VERL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VERL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.2}" \
  actor_rollout_ref.rollout.n="${VERL_ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${VERL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME:-cybertown_planning_drgrpo}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_0.6b_dr_grpo}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node="${VERL_N_GPUS_PER_NODE:-2}" \
  trainer.nnodes=1 \
  trainer.save_freq="${VERL_SAVE_FREQ:-10}" \
  trainer.test_freq="${VERL_TEST_FREQ:-5}" \
  trainer.total_epochs="${VERL_TOTAL_EPOCHS:-15}" \
  actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  reward_model.enable=False \
  reward_model.reward_manager="${VERL_REWARD_MANAGER:-GsiBatchRewardManager}" \
  reward_model.reward_loop_source="${VERL_REWARD_LOOP_SOURCE:-importlib}" \
  reward_model.reward_loop_module_path="${VERL_REWARD_LOOP_MODULE_PATH:-${REWARD_FILE_PATH}}" \
  reward_model.reward_loop_class_name="${VERL_REWARD_LOOP_CLASS_NAME:-GsiBatchRewardManager}" \
  reward_model.enable_resource_pool=False \
  custom_reward_function.path="${REWARD_FILE_PATH}" \
  custom_reward_function.name="${VERL_CUSTOM_REWARD_FN:-cybertown_score_fn_batched}" \
  "$@"
