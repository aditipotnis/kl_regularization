#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:59:59
#SBATCH --account=def-sreddy
#SBATCH --output=logs/slurm-%j.out
#SBATCH --mem 128G

# set up your environment
module load python/3.10 gcc arrow/18.1.0 StdEnv/2023
source venv/new-runs/bin/activate
unset ROCR_VISIBLE_DEVICES
kl=$1
penalty_type=$2

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
algorithm.adv_estimator=rloo \
algorithm.use_kl_in_reward=False \
algorithm.kl_ctrl.kl_coef=${kl} \
algorithm.kl_penalty=${penalty_type} \
data.train_files=./data/Deepscale-R_train.parquet \
data.val_files=./data/Deepscale-R_test.parquet \
data.train_batch_size=64 \
data.max_prompt_length=2048 \
data.max_response_length=8192 \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=${kl} \
actor_rollout_ref.actor.kl_loss_type="k3" \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.temperature=0.6 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.rollout.max_num_batched_tokens=12000 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
trainer.critic_warmup=0 \
trainer.logger=['console','wandb'] \
trainer.project_name='kl_investigation' \
trainer.experiment_name=${penalty_type}${kl} \
trainer.n_gpus_per_node=4 \
trainer.nnodes=1 \
trainer.save_freq=50 \
trainer.test_freq=5 \
trainer.total_epochs=15 \
trainer.validation_data_dir=/scratch/aditi22/verl/val_outputs/Deepscale-R-one-${penalty_type}${kl}/ \
trainer.save_freq=10 trainer.test_freq=10000 \
trainer.rollout_data_dir=/scratch/aditi22/verl/rollout_outputs/Deepscale-R-one-${penalty_type}${kl} \
trainer.default_local_dir=/scratch/aditi22/verl/checkpoints/Deepscale-R-one-${penalty_type}${kl} \
actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.val_kwargs.n=1 