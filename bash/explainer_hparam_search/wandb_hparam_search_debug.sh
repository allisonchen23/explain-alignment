#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory
#SBATCH -o slurm_logs/%j_%x_out.txt         # send stdout to outfile
#SBATCH -e slurm_logs/%j_%x_err.txt         # send stderr to errfile
#SBATCH -t 72:00:00           # time requested in hour:minute:second

CONFIG_PATH="configs/explainer_hparam_search/$1.json"
python explainer_hparam_search_wandb.py \
--config $CONFIG_PATH \
--train_script_path src/train_hparam_search.py \
--learning_rates 1e-4 1e-3 \
--weight_decays 0 1e-1 \
--build_save_dir