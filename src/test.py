import argparse
import torch
from tqdm import tqdm
import sys
import os
# sys.path.insert(0, 'src')

import datasets.datasets as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.utils import read_lists, ensure_dir
from parse_config import ConfigParser
from predict import predict


def main(config, test_data_loader=None):
    logger = config.get_logger('test')
    logger.info("Results saved to {}".format(os.path.dirname(config.log_dir)))

    # General arguments for data loaders
    dataset_args = config.config['dataset']['args']
    data_loader_args = config.config['data_loader']['args']
    # The architecture of the Edited model already normalizes
    if config.config['arch']['type'] == "ModelWrapperSanturkar":
        dataset_args['normalize'] = False
        logger.warning("Using edited model architecture. Overriding normalization for dataset to False.")

    # setup data_loader instances
    if test_data_loader is None:
        test_dataset = config.init_obj('dataset', module_data)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            **data_loader_args
        )
        logger.info("Created test ({} images) dataloader from {}.".format(
            len(test_dataset),
            test_dataset.dataset_dir))

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['type'], model.get_n_params()))

    # First priority is check for resumed path
    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        logger.info("Restored weights from {}".format(config.resume))
    elif model.get_checkpoint_path() != "":
        logger.info("Restored weights from {}".format(model.get_checkpoint_path()))
    else:
        raise ValueError("No checkpoint provided to restore model from")

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = []
    for met in config['metrics']:
        metric_fns.append(getattr(module_metric, met))
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Save results as a pickle file for easy deserialization
    metric_save_path = os.path.join(str(config.log_dir), 'test_metrics.pth')
    logits_save_path = os.path.join(str(config.log_dir), 'logits.pth')
    log = predict(
        data_loader=test_data_loader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device,
        log_save_path=metric_save_path,
        output_save_path=logits_save_path)
    for log_key, log_item in log.items():
        logger.info("{}: {}".format(log_key, log_item))
    # logger.info(log)
    logger.info("Saved test metrics to {}".format(metric_save_path))

    # Final message
    logger.info("Access results at {}".format(os.path.dirname(config.log_dir)))
    return log


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
