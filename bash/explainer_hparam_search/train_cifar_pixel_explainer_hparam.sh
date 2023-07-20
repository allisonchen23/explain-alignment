#!/bin/bash

python explainer_hparam_search.py \
--config configs/train_cifar_pixel_explainer_hparam.json \
--learning_rates 1e-4 1e-3 5e-2 1e-2 5e-1 1e-1 \
--weight_decays 0 1e-1 1e-2 1e-3 \
--build_save_dir