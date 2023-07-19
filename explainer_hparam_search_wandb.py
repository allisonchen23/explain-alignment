
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
from setup_cifar10 import setup_cifar10

def setup_dataloaders(config_json):
    dataset_args = config_json['dataset']['args']
    if 'cifar' in dataset_args['input_features_path']:
        train_descriptors_dataset = module_data.KDDataset(split='train', **dataset_args)
        test_descriptors_dataset = module_data.KDDataset(split='test', **dataset_args)
    elif 'ade20k' in dataset_args['input_features_path']:
        train_descriptors_dataset = module_data.KDDataset(split='train', **dataset_args)
        test_descriptors_dataset = module_data.KDDataset(split='val', **dataset_args)
    else:
        raise ValueError("Expected 'cifar' or 'ade20k' in input_features_path. Received {}".format(
            dataset_args['input_features_path']
        ))
    dataloader_args = config_json['data_loader']['args']
    train_descriptors_dataloader = torch.utils.data.DataLoader(
        train_descriptors_dataset,
        shuffle=True,
        **dataloader_args)
    test_descriptors_dataloader = torch.utils.data.DataLoader(
        test_descriptors_dataset,
        shuffle=False,
        **dataloader_args)
    return train_descriptors_dataloader, test_descriptors_dataloader

def restore_and_test(model,
                    config,
                    trial_dir,
                    model_restore_path,
                    val_dataloader,
                    metric_fns,
                    device,
                    loss_fn):

    model_restore_path = os.path.join(config.save_dir, 'model_best.pth')
    output_save_path = os.path.join(trial_dir, "val_outputs.pth")
    log_save_path = os.path.join(trial_dir, "val_metrics.pth")

    model.restore_model(model_restore_path)

    validation_data = predict(
        data_loader=val_dataloader,
        model=model,
        metric_fns=metric_fns,
        device=device,
        loss_fn=loss_fn,
        output_save_path=output_save_path,
        log_save_path=log_save_path)

    return validation_data

def save_best_outputs_predictions(best_trial_dir,
                                  save_best_model_dir,
                                  log_path,
                                  print_timestamp=True,
                                  model=None,
                                  val_dataloader=None,
                                  metric_fns=None,
                                  loss_fn=None,
                                  device=None):
    # Set flag for inconsistent results
    inconsistent_results = False
    # Copy config file
    config_path = os.path.join(best_trial_dir, 'models', 'config.json')
    copy_file(config_path, save_best_model_dir)

    # Validate that model run on the val_dataloader will give the same outputs as in test_outputs

    # Load test outputs
    test_outputs_path = os.path.join(save_best_model_dir, 'outputs.pth')
    test_outputs = torch.load(test_outputs_path)

    if model is not None and val_dataloader is not None:
        config_json = read_json(config_path)
        informal_log("Verifying saved model's performance is consistent...", log_path, timestamp=print_timestamp)

        # Obtain metric_fns, device, and loss_fn if not provided
        if metric_fns is None:
            metric_fns = [getattr(module_metric, met) for met in config_json['metrics']]
        if device is None:
            device, _ = prepare_device(config_json['n_gpu'])
        if loss_fn is None:
            loss_fn = getattr(module_loss, config_json['loss'])

        model_restore_path = os.path.join(save_best_model_dir, 'model.pth')
        model.restore_model(model_restore_path)

        validation_data = predict(
            data_loader=val_dataloader,
            model=model,
            metric_fns=metric_fns,
            device=device,
            loss_fn=loss_fn)

        val_metrics = validation_data['metrics']
        val_outputs = validation_data['logits']

        # Compare accuracy
        # Load hparams with validation accuracy in it
        hparams_path = os.path.join(save_best_model_dir, 'hparams.json')
        hparams = read_json(hparams_path)
        if 'val_acc' in hparams:
            accuracy = hparams['val_acc']

            if accuracy == val_metrics['accuracy'] and (test_outputs == val_outputs).all():
                informal_log("Model is consistent with saved outputs", log_path, timestamp=print_timestamp)
            if accuracy != val_metrics['accuracy']:
                informal_log("WARNING! Saved model inconsistent. Final validation on best model could not reproduce accuracy.", log_path, timestamp=print_timestamp)
                informal_log("Loaded {:.3f} from 'hparams.json' and obtained {:.3f} when running".format(
                    accuracy * 100, val_metrics['accuracy'] * 100), log_path, timestamp=print_timestamp)
                inconsistent_results = True
            if not (test_outputs == val_outputs).all():
                informal_log("WARNING! Saved model inconsistent. Final validation on best model could not reproduce same predictions.", log_path, timestamp=print_timestamp)
                inconsistent_results = True
        else:
            hparams['val_acc'] = val_metrics['accuracy']
        # Add n_params to hparams if it's not already there
        if 'n_params' not in hparams:
            hparams['n_params'] = int(model.get_n_params())
        print(hparams, type(hparams))
        write_json(hparams, hparams_path)

    # Calculate softmax probabilities & predictions
    test_probabilities = torch.softmax(test_outputs, dim=1)
    test_predictions = torch.argmax(test_outputs, dim=1)

    # Move all to CPU and convert to numpy
    test_outputs = test_outputs.cpu().numpy()
    test_probabilities = test_probabilities.cpu().numpy()
    test_predictions = test_predictions.cpu().numpy()

    outputs_predictions = {
        'test': {
            'outputs': test_outputs,
            'probabilities': test_probabilities,
            'predictions': test_predictions
        }
    }
    if inconsistent_results:
        outputs_predictions_save_path = os.path.join(save_best_model_dir, 'INCONSISTENT_outputs_predictions.pth')
    else:
        outputs_predictions_save_path = os.path.join(save_best_model_dir, 'outputs_predictions.pth')
    torch.save(outputs_predictions, outputs_predictions_save_path)

    return outputs_predictions_save_path


def run_hparam_search(config_json,
                      config_path,
                      learning_rates,
                      weight_decays,
                      debug=False,
                      print_timestamp=True,
                      ):
    if debug:
        config_json['trainer']['epochs'] = 1
        config_json['trainer']['save_dir'] = config_json['trainer']['save_dir'].replace('saved/', 'saved/debug/')

    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    log_path = os.path.join(config_json['trainer']['save_dir'], timestamp, 'log.txt')
    informal_log("Hyperparameter search", log_path, timestamp=print_timestamp)
    informal_log("Learning rates: {}".format(learning_rates), log_path, timestamp=print_timestamp)
    informal_log("Weight decays: {}".format(weight_decays), log_path, timestamp=print_timestamp)


    # Setup data loaders, device, loss, and metrics
    informal_log("Setting up train and test dataloaders...", log_path, timestamp=print_timestamp)
    # train_dataloader, test_dataloader = setup_dataloaders(config_json=config_json)
    device, device_ids = prepare_device(config_json['n_gpu'])
    metric_fns = [getattr(module_metric, met) for met in config_json['metrics']]
    loss_fn = getattr(module_loss, config_json['loss'])

    # Variable relevant for hparam search
    best = {
        'lr': -1,
        'wd': -1,
        'val_acc': -1
    }
    n_trials = len(learning_rates) * len(weight_decays)
    trial_idx = 1
    sweep_config = {
        'method': 'grid',
        'name': config_json['name'],
        'program': 'src/train.py',
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

        ]
    }
    sweep_id = wandb.sweep(
        sweep_config, 
        project=config_json['name']
    )
    print(sweep_id)
    for trial_idx in range(n_trials):
        wandb.agent('grad-student-descent/CIFAR_SIFT_explainer_hparam/{}'.format(sweep_id))
        # os.system('sbatch bash/start_sweep.sh {} {}'.format(sweep_id, config_json['name']))

    # for lr in learning_rates:
    #     for wd in weight_decays:
    #         # Update config json
    #         config_json['optimizer']['args'].update({
    #             'lr': lr,
    #             'weight_decay': wd
    #         })

    #         # Create run ID for trial
    #         informal_log("Trial {}/{}: LR = {} WD = {}".format(
    #             trial_idx, n_trials, lr, wd), log_path)
    #         run_id = os.path.join(timestamp, 'trials', 'lr_{}-wd_{}'.format(lr, wd))
    #         config = ConfigParser(config_json, run_id=run_id)

    #         # Train model
    #         informal_log("Training model...", log_path, timestamp=print_timestamp)

    #         model = train_fn(
    #             config=config,
    #             train_data_loader=train_dataloader,
    #             val_data_loader=test_dataloader)

    #         # Set paths for restoring model and trial
    #         model_restore_path = os.path.join(config.save_dir, 'model_best.pth')
    #         trial_dir = os.path.dirname(config.save_dir)
    #         # Run on validation set using predict function
    #         informal_log("Running model on test set...", log_path, timestamp=print_timestamp)
    #         validation_data = restore_and_test(
    #             model=model,
    #             config=config,
    #             trial_dir=trial_dir,
    #             model_restore_path=model_restore_path,
    #             val_dataloader=test_dataloader,
    #             metric_fns=metric_fns,
    #             device=device,
    #             loss_fn=loss_fn)

    #         val_accuracy = validation_data['metrics']['accuracy']
    #         informal_log("Test set accuracy: {:.3f}".format(val_accuracy * 100), log_path, timestamp=print_timestamp)

    #         # If best accuracy is achieved, print and save
    #         if val_accuracy > best['val_acc']:
    #             best.update({
    #                 'n_params': int(model.get_n_params()),
    #                 'lr': lr,
    #                 'wd': wd,
    #                 'val_acc': val_accuracy
    #             })
    #             informal_log("Best accuracy of {:.3f} with lr={} and wd={}".format(val_accuracy, lr, wd), log_path, timestamp=print_timestamp)
    #             informal_log("Trial path: {}".format(trial_dir), log_path, timestamp=print_timestamp)

    #             # Copy model and outputs to 1 directory for easy access
    #             best_save_dir = os.path.join(os.path.dirname(os.path.dirname(trial_dir)), 'best')
    #             ensure_dir(best_save_dir)
    #             best_outputs_save_path = os.path.join(best_save_dir, 'outputs.pth')
    #             best_model_save_path = os.path.join(best_save_dir, 'model.pth')
    #             best_hparam_save_path = os.path.join(best_save_dir, 'hparams.json')

    #             torch.save(validation_data['logits'], best_outputs_save_path)
    #             model.save_model(best_model_save_path)
    #             with open(best_hparam_save_path, "w") as f:
    #                 json.dump(best, f)
    #             informal_log("Saved model and outputs to {}".format(best_save_dir), log_path, timestamp=print_timestamp)
    #             # Save path to best trial_dir
    #             best_trial_dir = trial_dir

    #         informal_log("", log_path)

    #         trial_idx += 1

    # # Obtain outputs and predictions on validation data
    # outputs_predictions_save_path = save_best_outputs_predictions(
    #     best_trial_dir=best_trial_dir,
    #     save_best_model_dir=best_save_dir,
    #     log_path=log_path,
    #     print_timestamp=print_timestamp,
    #     model=model,
    #     val_dataloader=test_dataloader,
    #     metric_fns=metric_fns,
    #     loss_fn=loss_fn,
    #     device=device)
    # informal_log("Saved outputs & predictions to {}".format(outputs_predictions_save_path), log_path, timestamp=print_timestamp)


def build_save_dir(config_json, path_prefix='data/explainer_inputs'):
    '''
    Following format from generating the explainer inputs, the new path should be:
        root / dataset_type / input_type / <more params> / explainer hidden layers
    '''
    save_root = config_json['trainer']['save_dir']
    input_dataset_path = config_json['dataset']['args']['input_features_path']
    # Obtain relative path from the path prefix (typically 'data/explainer_inputs')
    relative_path = os.path.relpath(path=input_dataset_path, start=path_prefix)
    # Remove filename from path
    save_local_dir = os.path.dirname(relative_path)
    # input_dataset_name = os.path.basename(input_dataset_path).split("explainer_inputs.pth")[0]
    # save_local_dir = input_dataset_name.replace('_', '/')

    # Obtain number of hidden features
    hidden_layers = config_json['arch']['args']['n_hidden_features']
    if len(hidden_layers) == 0:
        hidden_string = 'hidden_NA'
    else:
        hidden_string = 'hidden'
        for h in hidden_layers:
            hidden_string += '_{}'.format(h)

    save_dir = os.path.join(save_root, save_local_dir, hidden_string)
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str, help='Path to config file')
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

    if args.build_save_dir:
        config_json['trainer']['save_dir'] = build_save_dir(config_json)

    run_hparam_search(
        config_json=config_json,
        config_path=args.config,
        learning_rates=args.learning_rates,
        weight_decays=args.weight_decays,
        debug=args.debug
    )
