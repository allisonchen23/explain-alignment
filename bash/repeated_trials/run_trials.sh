#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory 
#SBATCH -o slurm_logs/%A_%a_out.txt         # send stdout to outfile
#SBATCH -e slurm_logs/%A_%a_err.txt         # send stderr to errfile
#SBATCH -t 72:00:00           # time requested in hour:minute:second

CONFIG_PATH="configs/repeated_trials/$1.json"

echo $CONFIG_PATH
python src/train.py \
--config $CONFIG_PATH \
--trial_id $SLURM_ARRAY_TASK_ID