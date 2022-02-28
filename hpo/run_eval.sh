#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH --job-name v2_54_eval
#SBATCH -o logs/evaluation/%A-%a.%x.o
#SBATCH -e logs/evaluation/%A-%a.%x.e

#SBATCH --mail-user=ozturk@informatik.uni-freiburg.de
#SBATCH --mail-type=END,FAIL

#SBATCH --gres=gpu:1

#SBATCH -a 1-25

source /home/ozturk/anaconda3/bin/activate metadl
pwd

ARGS_FILE=hpo/evaluation_failure.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

echo $TASK_SPECIFIC_ARGS
python -m hpo.model $TASK_SPECIFIC_ARGS --task_id $SLURM_ARRAY_TASK_ID
