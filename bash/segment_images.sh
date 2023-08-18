#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory 
#SBATCH -o slurm_logs/%A_%a_run.txt         # send stdout to outfile
#SBATCH -e slurm_logs/%A_%a_err.txt         # send stderr to errfile
#SBATCH -t 72:00:00           # time requested in hour:minute:second

python src/ace/segment_images.py \
--mode features \
--image_labels_path data/ade20k/full_ade20k_imagelabels.pth \
--save_dir data/ade20k/ace \