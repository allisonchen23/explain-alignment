#!/bin/bash
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory 
#SBATCH -o slurm_logs/%x_%i_out.txt         # send stdout to outfile
#SBATCH -e slurm_logs/%x_%i_err.txt         # send stderr to errfile
#SBATCH -t 72:00:00           # time requested in hour:minute:second

python explainer_hparam_search_wandb.py \
--config configs/explainer_hparam_search/train_ade20k_dense_CAV_182_concept_explainer_hparam.json \
--train_script_path src/train_hparam_search.py \
--learning_rates 1e-4 1e-3 5e-2 1e-2 5e-1 1e-1 \
--weight_decays 0 1e-1 1e-2 1e-3 \
--build_save_dir