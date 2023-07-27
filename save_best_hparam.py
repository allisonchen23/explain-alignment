import os, sys
import argparse
import torch

sys.path.insert(0, 'src')
import datasets.datasets as module_data
from utils.utils import read_json, ensure_dir
from utils.model_utils import prepare_device
import model.metric as module_metric
import model.loss as module_loss
import model.model as module_model
from explainer_hparam_search import save_best_outputs_predictions

# Save best outputs after hparam search
# DON"T DELETE THIs YET

# best_trial_dir = 'saved/PlacesCategoryClassification/0510_102912/ADE20K_predictions/saga/KD_baseline_explainer/hparam_search/0523_164052/trials/lr_0.001-wd_0'
# save_best_model_dir = 'saved/PlacesCategoryClassification/0510_102912/ADE20K_predictions/saga/KD_baseline_explainer/hparam_search/0523_164052/best'
def copy_best_trial(best_trial_dir,
                    save_best_model_dir):
    config_path = os.path.join(best_trial_dir, 'models', 'config.json')
    config_json = read_json(config_path)

    # Dataloader
    print("Setting up dataloader")
    dataset_args = config_json['dataset']['args']
    val_dataset = module_data.KDDataset(split='test', **dataset_args)
    dataloader_args = config_json['data_loader']['args']
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_args)

    # Model
    print("Setting up model")
    arch = config_json['arch']['type']
    model_args = config_json['arch']['args']
    model = getattr(module_model, arch)(**model_args)

    # Device,, metrics, and loss
    print("Setting up device, metric, and loss")
    device, device_ids = prepare_device(config_json['n_gpu'])
    metric_fns = [getattr(module_metric, met) for met in config_json['metrics']]
    loss_fn = getattr(module_loss, config_json['loss'])

    # Move model to device
    model = model.to(device)

    save_best_outputs_predictions(
        best_trial_dir=best_trial_dir,
        save_best_model_dir=save_best_model_dir,
        log_path=None,
        model=model,
        val_dataloader=val_dataloader,
        metric_fns=metric_fns,
        loss_fn=loss_fn,
        device=device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_dir', required=True, help='Path to directory where \'models\' dir is')
    parser.add_argument('--save_best_model_dir', required=False, default=None, help='Optional path to where to save best model to. Default is in grandparent directory/best')
    args = parser.parse_args()
    
    if args.save_best_model_dir is None:
        args.save_best_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.trial_dir)),
            'best')
    ensure_dir(args.save_best_model_dir)
    
    copy_best_trial(
        best_trial_dir=args.trial_dir,
        save_best_model_dir=args.save_best_model_dir)