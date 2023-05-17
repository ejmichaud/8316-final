#!/bin/bash
#SBATCH --job-name=train-0
#SBATCH --ntasks=1
#SBATCH --time=0-1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/om/user/ericjm/results/class/8.316/final/train-0/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/class/8.316/final/train-0/logs/slurm-%A_%a.err
#SBATCH --mem=5GB
#SBATCH --array=0-53

python /om2/user/ericjm/class/8.316/final/experiments/train-0/eval.py $SLURM_ARRAY_TASK_ID

