import os, sys
from sklearn.cluster import KMeans
import argparse
from datetime import datetime
import torch
import numpy as np

sys.path.insert(0, 'src')
from utils.visualizations import plot
from utils.utils import informal_log, ensure_dir
# debug = True
# step_size = 2
def dense_sift_kmeans_search(debug, step_size, sigma, 
                             image_size=(32, 32)):
    # Load descriptors

    # if dense:
    sift_data_path = os.path.join("saved", "cifar10", 
                              'sift_{}_{}_sigma{}'.format(image_size[0], image_size[1], sigma), 
                              'dense_stride_{}'.format(step_size),
                              'sift_keypoints_descriptors.pth')
    # else:
    #     sift_data_path = os.path.join(cifar10_save_dir, 
    #                                   'sift_{}_{}_sigma{}'.format(image_size[0], image_size[1], sigma), 
    #                                   'sift_keypoints_descriptors.pth')
    save_dir = os.path.dirname(sift_data_path)
    log_path = os.path.join(save_dir, 'sift_kmeans_log.txt')
    informal_log("KMeans hyperparameter search", log_path)

    
    informal_log("[{}] Loading features from {}".format(
        datetime.now().strftime(r'%m%d_%H%M%S'), sift_data_path), log_path)
    sift_data = torch.load(sift_data_path)

    train_descriptors = sift_data['train']['descriptors']
    flat_train_descriptors = np.concatenate(train_descriptors, axis=0)
    informal_log("There are {} descriptors of size {}".format(
        flat_train_descriptors.shape[0], flat_train_descriptors.shape[1]), log_path)

    if debug:
        ks = [5]
    else:
        ks = [500, 1000, 1500, 2000, 2500]
    n_init = 10
    inertias = []


    for k in ks:
        informal_log("[{}] Calculating k-means for k={}".format(
            datetime.now().strftime(r'%m%d_%H%M%S'), k), log_path)
        if debug:
            kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=5)
        else:
            kmeans = KMeans(n_clusters=k, n_init=n_init)
        kmeans = kmeans.fit(flat_train_descriptors)
        informal_log("[{}] Inertia: {}".format(
            datetime.now().strftime(r'%m%d_%H%M%S'), kmeans.inertia_), log_path)
        inertias.append(kmeans.inertia_)
        kmeans_save_path = os.path.join(os.path.dirname(sift_data_path),
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
    parser.add_argument('--sigma', type=float, required=True)
    
    args = parser.parse_args()
    dense_sift_kmeans_search(
        debug=args.debug,
        step_size=args.step_size,
        sigma=args.sigma)
    
    