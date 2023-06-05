import os, sys
from sklearn.cluster import KMeans, MiniBatchKMeans
import argparse
from datetime import datetime
import torch
import numpy as np

sys.path.insert(0, 'src')
from utils.visualizations import plot
from utils.utils import informal_log, ensure_dir
# debug = True
# step_size = 2

seed = 0
np.random.seed(seed)

def dense_sift_kmeans_search(debug, step_size, sigma, 
                             n_samples=None,
                             image_size=(32, 32),
                             ks=[500, 1000, 1500, 2000, 2500],
                             mini_batch_size=1024):
    sift_data_path = os.path.join("saved", "cifar10", 
                              'sift_{}_{}_sigma{}'.format(image_size[0], image_size[1], sigma), 
                              'dense_stride_{}'.format(step_size),
                              'sift_keypoints_descriptors.pth')
    timestamp = datetime.now().strftime(r"%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(sift_data_path), timestamp)
    
    log_path = os.path.join(save_dir, 'sift_kmeans_log.txt')
    informal_log("KMeans hyperparameter search", log_path)

    # Load descriptors
    # informal_log("[{}] Loading features from {}".format(
    #     datetime.now().strftime(r'%m%d_%H%M%S'), sift_data_path), log_path)
#     sift_data = torch.load(sift_data_path)

#     train_descriptors = sift_data['train']['descriptors']
#     n_data = len(train_descriptors)

    # Randomly sample if needed to reduce number of examples
    if n_samples is not None:
        sampled_descriptor_save_path = os.path.join(save_dir, '{}_train_sift_keypoint_descriptors.pth'.format(n_samples))
        if os.path.exists(sampled_descriptor_save_path):
            informal_log("[{}] Loading features from {}".format(
            datetime.now().strftime(r'%m%d_%H%M%S'), sampled_descriptor_save_path), log_path)
            train_descriptors = torch.load(sampled_descriptor_save_path)
        else:
            informal_log("[{}] Loading features from {}".format(
                datetime.now().strftime(r'%m%d_%H%M%S'), sift_data_path), log_path)
            sift_data = torch.load(sift_data_path)
            train_descriptors = sift_data['train']['descriptors']
            n_data = len(train_descriptors)
            if n_samples >= n_data:
                raise ValueError("N_samples must be less than n_data ({} and {} respectively)".format(
                    n_samples, n_data))
            random_indices = np.random.choice(n_data, 
                                      size=n_samples, 
                                      replace=False)
            train_descriptors = np.stack(train_descriptors, axis=0)
            train_descriptors = train_descriptors[random_indices, ...]

            torch.save(train_descriptors, sampled_descriptor_save_path)
            indices_save_path = os.path.join(save_dir, "{}_train_indices.pth".format(n_samples))
            torch.save(random_indices, indices_save_path)
            informal_log("[{}] Sampled {} images and saving descriptors to {} and indices to {}".format(
                datetime.now().strftime(r'%m%d_%H%M%S'), 
                n_samples,
                sampled_descriptor_save_path,
                indices_save_path), log_path)
        
        # Flatten descriptors for clustering
        descriptor_dim = train_descriptors.shape[-1]
        flat_train_descriptors = train_descriptors.reshape((-1, descriptor_dim))
    else:  # No sampling
        informal_log("[{}] Loading features from {}".format(
            datetime.now().strftime(r'%m%d_%H%M%S'), sift_data_path), log_path)
        sift_data = torch.load(sift_data_path)
        train_descriptors = sift_data['train']['descriptors']
        flat_train_descriptors = np.concatenate(train_descriptors, axis=0)
    informal_log("There are {} descriptors of size {}".format(
        flat_train_descriptors.shape[0], flat_train_descriptors.shape[1]), log_path)

    if debug:
        ks = [5]
        
    n_init = 10
    inertias = []
    # Run KMeans for each
    for k in ks:
        informal_log("[{}] Calculating k-means for k={}".format(
            datetime.now().strftime(r'%m%d_%H%M%S'), k), log_path)
        if mini_batch_size > 0:
            if debug:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    n_init=n_init,
                    batch_size=mini_batch_size,
                    max_iter=5)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    n_init=n_init,
                    batch_size=mini_batch_size)
        if debug:
            kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=5)
        else:
            kmeans = KMeans(n_clusters=k, n_init=n_init)
        kmeans = kmeans.fit(flat_train_descriptors)
        informal_log("[{}] Inertia: {}".format(
            datetime.now().strftime(r'%m%d_%H%M%S'), kmeans.inertia_), log_path)
        inertias.append(kmeans.inertia_)
        kmeans_save_path = os.path.join(save_dir,
                                    '{}means'.format(k), 
                                    'descriptor_kmeans.pth')
        ensure_dir(os.path.dirname(kmeans_save_path))

        if os.path.exists(kmeans_save_path):
            kmeans_save_path = os.path.join(os.path.dirname(kmeans_save_path), 'descriptor_kmeans_new.pth')
        torch.save(kmeans, kmeans_save_path)
        informal_log("Saved clustering to {}".format(kmeans_save_path), log_path)
    informal_log("[{}] Ks: {} Inertias: {}".format(
        datetime.now().strftime(r'%m%d_%H%M%S'), ks, inertias), log_path)
    plot_save_path = os.path.join(save_dir, 'inertia_k.png')
    plot(
        xs=ks,
        ys=inertias,
        xlabel='K',
        ylabel='Inertia',
        save_path=plot_save_path)
    informal_log("[{}] Saved ks vs inertias plot to {}".format(
        datetime.now().strftime(r'%m%d_%H%M%S'), plot_save_path), log_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--step_size', type=int, required=True)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--mini_batch_size', type=int, default=-1)
    parser.add_argument('--ks', nargs='+', type=int, default=[500, 1000, 1500, 2000, 2500])
    
    args = parser.parse_args()
    dense_sift_kmeans_search(
        debug=args.debug,
        step_size=args.step_size,
        n_samples=args.n_samples,
        sigma=args.sigma,
        ks=args.ks,
        mini_batch_size=args.mini_batch_size)
    
    