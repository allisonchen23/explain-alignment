#!/bin/bash

python explainer_hparam_search_wandb.py \
--config configs/wandb_train_cifar_SIFT_explainer_hparam.json \
--train_script_path src/train_hparam_search.py \
--learning_rates 1e-4 1e-3 5e-2 1e-2 5e-1 1e-1 \
--weight_decays 0 1e-1 1e-2 1e-3 \