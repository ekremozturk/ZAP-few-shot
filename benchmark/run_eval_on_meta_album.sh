#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH --job-name MA_Ev
#SBATCH -o logs/meta_album_evaluation/%A-%a.%x.o
#SBATCH -e logs/meta_album_evaluation/%A-%a.%x.e

#SBATCH --mail-user=ozturk@informatik.uni-freiburg.de
#SBATCH --mail-type=END,FAIL

#SBATCH --gres=gpu:1

#SBATCH -a 1-1

source /home/ozturk/anaconda3/bin/activate metadl
pwd

ARGS_FILE=benchmark/meta_album.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

echo $TASK_SPECIFIC_ARGS
python -m benchmark.run_local_test $TASK_SPECIFIC_ARGS --task_id $SLURM_ARRAY_TASK_ID
