import argparse
import collections
import torch
import numpy as np
import os, sys
import wandb
sys.path.insert(0, 'src')
# import data_loader.data_loaders as module_data
import datasets.datasets as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer.trainer import Trainer
from utils.utils import read_lists, read_json
from utils.model_utils import prepare_device

from parse_config import ConfigParser
from predict import predict


def main(config_json, timestamp=None, train_data_loader=None, val_data_loader=None, seed=0):
    # Code to set up config file if part of wandb sweep
    try:
        wandb.init()
        wandb_config = wandb.config
        if 'lr' in wandb_config:
            config_json['optimizer']['args']['lr'] = wandb_config.lr
        if 'wd' in wandb_config:
            config_json['optimizer']['args']['weight_decay'] = wandb_config.wd
        run_id = build_run_id(config_json)
        if timestamp is not None:
            run_id = os.path.join(run_id, timestamp)
        run_id = os.path.join(
            run_id,
            'trials',
            'lr_{}-wd_{}'.format(wandb_config.lr, wandb_config.wd)
        )
        config = ConfigParser(config_json, run_id=run_id)
    except:
        print("no config")
        config = ConfigParser(config_json)
        wandb.init(
            project=config.config['name'],
            name=config.run_id,
            config={
                'arch': config.config['arch']['type'],
                'lr': config.config['optimizer']['args']['lr'],
                'wd': config.config['optimizer']['args']['weight_decay'],
                'optimizer': config.config['optimizer']['type'],
                'save_dir': os.path.dirname(config.save_dir)
                
            }
        )

    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
    logger = config.get_logger('train')

    # setup data_loader instances
    if train_data_loader is None and val_data_loader is not None:
        raise ValueError("No data loader passed for validation")
    elif train_data_loader is not None and val_data_loader is None:
        raise ValueError("No data loader passed for training")
    elif train_data_loader is None and val_data_loader is None:
        # General arguments for data loaders
        data_loader_args = config.config['data_loader']['args']

        # Create train data loader
        if 'train_split' in config.config['dataset']:
            dataset = config.init_obj('dataset', module_data)
            train_split = config.config['dataset']['train_split']
            assert train_split > 0 and train_split < 1, "Invalid value for train_split: {}. Must be [0, 1]".format(train_split)
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset,
                [train_split, 1 - train_split],
                generator=torch.Generator().manual_seed(seed))

            logger.info("Dataset path(s): \n\t{}".format(dataset.root))
        # Create train and val datasets separately
        elif config.config['dataset']['type'] in ['KDDataset', 'CIFAR10TorchDataset']:
            train_dataset = config.init_obj('dataset', module_data, split='train')
            try:
                val_dataset = config.init_obj('dataset', module_data, split='val')
            except:
                val_dataset = config.init_obj('dataset', module_data, split='test')
            train_split = len(train_dataset) / (len(train_dataset) + len(val_dataset))
            # logger.info("Dataset path(s): \n\t{}\n\t{}".format(
            #     train_dataset.input_features_path,
            #     train_dataset.labels_path))
        else:
            raise ValueError("Dataset type '{}' not supported".format(config.config['dataset']['type']))
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            **data_loader_args
        )
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            **data_loader_args
        )

        logger.info("Created train ({} images) and val ({} images) datasets with {}/{} split.".format(
                len(train_dataset),
                len(val_dataset),
                train_split,
                1 - train_split,
            ))

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    try:
        logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['args']['type'], model.get_n_params()))
    except:
        logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['type'], model.get_n_params()))
    if model.get_checkpoint_path() != "":
        logger.info("Restored weights from {}".format(model.get_checkpoint_path()))
    else:
        logger.info("Training from scratch.")

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    if config.config['lr_scheduler']['type'] != "None":
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    
    if 'save_val_results' in config.config['trainer'] and \
    config.config['trainer']['save_val_results']:
        val_metric_save_path = os.path.join(os.path.dirname(config.save_dir), 'val_metrics.pth')
        val_outputs_save_path = os.path.join(os.path.dirname(config.save_dir), 'val_outputs.pth')
        predict(
            data_loader=val_data_loader,
            model=model,
            metric_fns=metrics,
            device=device,
            loss_fn=criterion,
            output_save_path=val_outputs_save_path,
            log_save_path=val_metric_save_path)
        print("Saving validation results to {}".format(os.path.dirname(val_metric_save_path)))
    
    return model

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

def build_run_id(config_json, path_prefix='data/explainer_inputs'):
    '''
    Following format from generating the explainer inputs, the new path should be:
        root / dataset_type / input_type / <more params> / explainer hidden layers
    '''
    # save_root = config_json['trainer']['save_dir']
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
        run_id = 'hidden_NA'
    else:
        run_id = 'hidden'
        for h in hidden_layers:
            run_id += '_{}'.format(h)

    run_id = os.path.join(save_local_dir, run_id)
    return run_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--timestamp', default=None, type=str,
                      help='timestamp ID')
    # parser.add_argument('--build_save_dir', default=False, action='store_true')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--wd', '--weight_decay'], type=float, target='optimizer;args;wd'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--name'], type=str, target='name')
    ]
    args = parser.parse_args()
    config_json = read_json(args.config)
    # if args.build_save_dir:
    #     config_json['trainer']['save_dir'] = build_save_dir(config_json)

    # config = ConfigParser.from_args(args, options)
    # config = ConfigParser(config_json)
    main(config_json, timestamp=args.timestamp)
