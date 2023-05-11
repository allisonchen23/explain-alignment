'''
Test to see if the linear torch layer does any better than sklearn logistic regression
for training 16-way category classifier on top of ResNet18 backbone trained on Places365.

Conclusion: not really

Accuracy of torch.linear:
    0.6532191780821918
Accuracy of logistic regression with liblinear solver, l2 reg, lr = 0.01:
    0.6532191780821918
'''
import os, sys
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as Preprocessing
import argparse

sys.path.insert(0, 'src')
from utils.places365_pred_utils import get_class_category_dict
from utils.utils import ensure_dir, read_json
from utils.model_utils import prepare_device
from datasets.datasets import ImageDataset
from trainer.trainer import Trainer
import model.metric as module_metric
import model.loss as module_loss
from predict import predict
from parse_config import ConfigParser


class FeaturesDataset(Dataset):
    def __init__(self,
                 features,
                 labels=None):

        if labels is not None:
            assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

    def __len__(self):
        return len(self.features)

def train(config_path):
    config_dict = read_json(config_path)

    data_path = config_dict['paths']['data_path']
    train_features_path = config_dict['paths']['train_features_path']
    val_features_path = config_dict['paths']['val_features_path']

    # Load labels
    data = torch.load(data_path)
    train_paths = data['val_train']
    val_paths = data['val_val']
    category_labels = data['scene_category_labels']
    train_category_labels = [category_labels[path] for path in train_paths]
    val_category_labels = [category_labels[path] for path in val_paths]

    # Load features
    train_features_dict = torch.load(train_features_path)
    val_features_dict = torch.load(val_features_path)

    # Sanity checks for elementwise correspondence
    for idx, path in enumerate(train_features_dict['paths']):
        assert path == train_paths[idx]
    for idx, path in enumerate(val_features_dict['paths']):
        assert path == val_paths[idx]

    train_features = train_features_dict['features']
    val_features = val_features_dict['features']

    # Length sanity checks
    assert len(train_features) == len(train_category_labels)
    assert len(val_features) == len(val_category_labels)

    print("Loaded {} samples for training and {} samples for validation".format(
        len(train_features), len(val_features)
    ))

    feature_dim = train_features[0].shape[0]
    n_classes = 16
    device, _ = prepare_device(config_dict['n_gpu'])
    linear_layer = torch.nn.Linear(
        in_features=feature_dim,
        out_features=n_classes,
        device=device)


    trainable_params = filter(lambda p: p.requires_grad, linear_layer.parameters())
    optimizer_type = config_dict['optimizer']['type']
    optimizer_args = config_dict['optimizer']['args']
    optimizer = getattr(torch.optim, optimizer_type)(trainable_params, **optimizer_args)
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params, **optimizer_args)

    loss = getattr(module_loss, config_dict['loss'])
    metrics = [getattr(module_metric, met) for met in config_dict['metrics']]

    lr_scheduler = None

    # Create datasets
    train_dataset = FeaturesDataset(
    features=train_features,
    labels=train_category_labels)

    val_dataset = FeaturesDataset(
        features=val_features,
        labels=val_category_labels)

    dataloader_args = config_dict['data_loader']['args']

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_args)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_args)

    config = ConfigParser(config_dict)

    trainer = Trainer(
        model=linear_layer,
        criterion=loss,
        metric_ftns=metrics,
        optimizer=optimizer,
        config=config,
        device=device,
        data_loader=train_dataloader,
        valid_data_loader=val_dataloader)

    trainer.train()

def test(config_path):
    config_dict = read_json(config_path)

    # data_path = config_dict['paths']['data_path']
    split = config_dict['paths']['split']
    features_path = config_dict['paths']['features_path'].format(split)

    # Load features & create data loader
    features_dict = torch.load(features_path)
    features = features_dict['features']
    print("Loaded {} samples for prediction".format(
        len(features)
    ))
    dataset = FeaturesDataset(
        features=features,
        labels=None)

    dataloader_args = config_dict['data_loader']['args']
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        **dataloader_args)

    # Create model
    feature_dim = features[0].shape[0]
    n_classes = 16
    device, _ = prepare_device(config_dict['n_gpu'])
    linear_layer = torch.nn.Linear(
        in_features=feature_dim,
        out_features=n_classes,
        device=device)

    restore_path = config_dict['arch']['restore_path']
    state_dict = torch.load(restore_path)['state_dict']
    linear_layer.load_state_dict(state_dict)

    linear_layer.eval()
    linear_layer = linear_layer.to(device)

    # Run model
    outputs = []
    with torch.no_grad():
        for idx, cur_features in tqdm(enumerate(dataloader)):
            cur_features = cur_features.to(device)
            output = linear_layer(cur_features)
            outputs.append(output)

    outputs = torch.cat(outputs, dim=0)
    probabilities = torch.softmax(outputs, dim=1)
    assert probabilities.shape == outputs.shape
    predictions = torch.argmax(outputs, dim=1)

    save_data = {
        'outputs': outputs.cpu().numpy(),
        'probabilities': probabilities.cpu().numpy(),
        'predictions': predictions.cpu().numpy()
    }

    ensure_dir(config_dict['save_dir'])
    save_path = os.path.join(config_dict['save_dir'], '{}_outputs_predictions.pth'.format(split))
    torch.save(save_data, save_path)
    print("Saved predictions and outputs for {} split to {}".format(split, save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=os.path.join('configs', 'train_features_scenecategory.json'), type=str,
                      help='config file path (default: None)')

    args = parser.parse_args()
    config_dict = read_json(args.config)
    method = config_dict['method']
    if method == 'train':
        train(args.config)
    elif method == 'test':
        test(args.config)
    else:
        raise ValueError("Unsupported method '{}'".format(method))