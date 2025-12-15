#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --account=def-sreddy
#SBATCH --output=logs/diversity_%j.out
#SBATCH --mem 128G

# set up your environment
module load python/3.10 gcc arrow/18.1.0 StdEnv/2023
source /home/aditi22/scratch/verl/diversity/bin/activate
unset ROCR_VISIBLE_DEVICES
cd /home/aditi22/scratch/verl
python compute_diversity.py \
  "rollout_outputs/Deepscale-R-one-kl0/1.jsonl" \
  --field output \
  --model-name BAAI/bge-m3 \
  --batch-size 8 \
  --device cuda \
  --max-texts 1000