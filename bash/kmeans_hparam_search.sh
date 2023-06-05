#!/bin/bash

python experimental_scripts/dense_sift_kmeans_search.py \
--step_size 2 \
--sigma 1.6 \
--mini_batch_size 2048 \
--ks 100 500 1000 1500 2000 \
