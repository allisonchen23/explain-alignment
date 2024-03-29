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
from predict import predict, restore_and_test


def main(config, train_data_loader=None, val_data_loader=None, seed=0):
    logger = config.get_logger('train')
    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        logger.info("Set seed to {}".format(seed))
    else:
        logger.info("No seed set")
        
    wandb.init(
            project=config.config['name'],
            name=config.run_id,
            config={
                'arch': config.config['arch']['type'],
                'lr': config.config['optimizer']['args']['lr'],
                'wd': config.config['optimizer']['args']['weight_decay'],
                'momentum': config.config['optimizer']['args']['momentum'],
                'optimizer': config.config['optimizer']['type'],
                'save_dir': os.path.dirname(config.save_dir),
                'seed': seed
            }
        )

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

            # Try to get test dataset
            try: 
                test_dataset = config.init_obj('dataset', module_data, split='test')
                logger.info("Loaded test dataset with {} samples".format(len(test_dataset)))
            except:
                test_dataset = None
                logger.info("No test dataset to load")
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

        if test_dataset is None:
            test_data_loader = None
        else:
            test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                shuffle=False,
                **data_loader_args
            )
        # Dataset logging
        if test_data_loader is None:
            train_split = len(train_dataset) / (len(train_dataset) + len(val_dataset))
            logger.info("Created train ({} images) and val ({} images) datasets with {}/{} split.".format(
                    len(train_dataset),
                    len(val_dataset),
                    train_split,
                    1 - train_split,
                ))
        else:
            n_total_images = len(train_dataset) + len(val_dataset) + len(test_dataset)
            logger.info("Created train ({} images), val ({} images), and test ({} images) dataloaders with {:.2f}/{:.2f}/{:.2f} split.".format(
                len(train_dataset),
                len(val_dataset),
                len(test_dataset),
                len(train_dataset) / n_total_images,
                len(val_dataset) / n_total_images,
                len(test_dataset) / n_total_images
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
        # Restore and save predictions from the best output
        model_restore_path = os.path.join(config.save_dir, 'model_best.pth')
        trial_dir = os.path.dirname(config.save_dir)
        val_metric_save_path = os.path.join(os.path.dirname(config.save_dir), 'val_metrics.pth')
        val_output_save_path = os.path.join(os.path.dirname(config.save_dir), 'val_outputs.pth')
        
        validation_data = restore_and_test(
            model=model,
            config=config,
            trial_dir=trial_dir,
            model_restore_path=model_restore_path,
            dataloader=val_data_loader,
            metric_fns=metrics,
            device=device,
            loss_fn=criterion,
            output_save_path=val_output_save_path,
            metric_save_path=val_metric_save_path
        )
        logger.info("Best model accuracy on validation set: {:.4f}".format(
            validation_data['metrics']['accuracy']
        ))
        logger.info("Saving validation predictions and results to {}".format(os.path.dirname(val_output_save_path)))
    
    if test_data_loader is not None:
        test_metric_save_path = os.path.join(os.path.dirname(config.save_dir), 'test_metrics.pth')
        test_output_save_path = os.path.join(os.path.dirname(config.save_dir), 'test_outputs.pth')
        test_data = restore_and_test(
            model=model,
            config=config,
            trial_dir=trial_dir,
            model_restore_path=model_restore_path,
            dataloader=test_data_loader,
            metric_fns=metrics,
            device=device,
            loss_fn=criterion,
            output_save_path=test_output_save_path,
            metric_save_path=test_metric_save_path
        )
        logger.info("Best model accuracy on test set: {:.4f}".format(
            test_data['metrics']['accuracy']
        ))
        logger.info("Saving test predictions and results to {}".format(os.path.dirname(test_output_save_path)))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('-s', '--seed', default=None, type=int,
                      help='Set seed (if not None)')
    parser.add_argument('--trial_id', default=None, type=str,
                      help='ID of trial if replacing timestamp')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--name'], type=str, target='name')
    ]
    args = parser.parse_args()
    config_json = read_json(args.config)
    config = ConfigParser(config_json, run_id=args.trial_id)
    # config = ConfigParser.from_args(parser, options)
    main(config,
         seed=args.seed)
