#!/bin/bash
#SBATCH --job-name=all-0
#SBATCH --ntasks=1
#SBATCH --time=0-4:00:00
#SBATCH --output=/om/user/ericjm/results/class/8.316/final/all-0/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/class/8.316/final/all-0/logs/slurm-%A_%a.err
#SBATCH --mem=3GB
#SBATCH --array=402,403,406,407,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499

python /om2/user/ericjm/class/8.316/final/experiments/all-0/eval.py $SLURM_ARRAY_TASK_ID

