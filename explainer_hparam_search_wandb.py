
import argparse
import json
import torch
import numpy as np
import os, sys
import shutil
import pickle
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from datetime import datetime
import wandb

sys.path.insert(0, 'src')
from utils.utils import ensure_dir, read_json, informal_log, copy_file, write_json
from utils.visualizations import plot
from utils.model_utils import prepare_device

import model.metric as module_metric
import model.loss as module_loss
import datasets.datasets as module_data
import model.model as module_model

from src.train import main as train_fn
from predict import predict
from parse_config import ConfigParser


sys.path.insert(0, 'setup')


def run_hparam_search(config_json,
                      config_path,
                      learning_rates,
                      weight_decays,
                      train_script_path,
                      debug=False,
                    #   print_timestamp=True,
                      ):
    # if debug:
    #     config_json['trainer']['epochs'] = 1
    #     config_json['trainer']['save_dir'] = config_json['trainer']['save_dir'].replace('saved/', 'saved/debug/')

    timestamp = str(datetime.now().strftime(r'%m%d_%H%M%S'))
    n_trials = len(learning_rates) * len(weight_decays)
    sweep_config = {
        'method': 'grid',
        'name': config_json['name'],
        'program': train_script_path,
        'metric': {
            'goal': 'maximize',
            'name': 'val/accuracy',
        },
        'parameters': {
            'lr': {'values': learning_rates},
            'wd': {'values': weight_decays}
        },
        'command': [
            'python',
            '${program}',
            '--config',
            config_path,
            '--trial_id',
            str(timestamp)

        ]
    }
    sweep_id = wandb.sweep(
        sweep_config, 
        project=config_json['name']
    )
    print(sweep_id)
    for trial_idx in range(n_trials):
        wandb.agent('grad-student-descent/{}/{}'.format(config_json['name'], sweep_id))
        # os.system('sbatch bash/start_sweep.sh {} {}'.format(sweep_id, config_json['name']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str, help='Path to config file')
    parser.add_argument('--train_script_path', required=True, type=str, help='Path to training script')
    parser.add_argument('--learning_rates', '--lr', type=float, nargs="+",
        default=[1e-4, 1e-3, 5e-2, 1e-2, 5e-1, 1e-1], help='Space delimited list of learning rates')
    parser.add_argument('--weight_decays', '--wd', type=float, nargs="+",
        default=[0, 1e-1, 1e-2, 1e-3], help="Space delimited list of weight decays")
    parser.add_argument('--build_save_dir', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()

    if args.debug:
        args.learning_rates = [1e-3]
        args.weight_decays = [0]

    config_json = read_json(args.config)

    run_hparam_search(
        config_json=config_json,
        config_path=args.config,
        train_script_path=args.train_script_path,
        learning_rates=args.learning_rates,
        weight_decays=args.weight_decays,
        debug=args.debug
    )
