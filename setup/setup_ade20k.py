import csv
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split
import pickle
import argparse
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, 'src')
from utils.utils import ensure_dir
from util

BRODEN_SUBSETS = ['ade20k', 'dtd', 'pascal']


def save_broden_subset(subset_name: str,
                       broden_path: str,
                       save_dir: str):

    # Check valid subset name
    assert subset_name in BRODEN_SUBSETS, "Subset '{}' not found in available subsets: {}".format(subset_name, BRODEN_SUBSETS)

    save_path = os.path.join(save_dir, '{}_imagelabels.pth'.format(subset_name))
    if os.path.exists(save_path):
        print("Path {} already exists. Aborting.".format(save_path))
        return

    print("Will save {} subset of Broden dataset from '{}' to '{}'".format(
        subset_name,
        broden_path,
        save_dir
    ))

    index_path = os.path.join(broden_path, 'index.csv')
    image_df = pd.read_csv(index_path)

    # Define variables
    images = []
    labels = {}
    scene_labels = {}

    n_images_broden = len(image_df)
    # Only save rows where image is from subset
    image_df = image_df[image_df['image'].str.contains(subset_name)]
    # Only save rows with an annotation for scene
    image_df = image_df[~image_df['scene'].isna()]
    n_images = len(image_df)
    print("Filtered from {} to {} images from {} with annotations".format(
        n_images_broden, n_images, subset_name))
    # Iterate through all images in index.csv
    for idx in tqdm(image_df.index):
        # if (image_df['image'][idx]).split('/')[0] != subset_name or pd.isna(pd.isna(image_df['scene'][idx])):
        #     continue
        full_image_name = os.path.join(broden_path, 'images/{}'.format(image_df['image'][idx]))
        #print(idx)
        images.append(full_image_name)
        labels[full_image_name] = []
        scene_labels[full_image_name] = image_df['scene'][idx]

        # Add part and object annotations
        for cat in ['object', 'part']:
            if image_df[cat].notnull()[idx]:
                for x in image_df[cat][idx].split(';'):
                    image_path = os.path.join(broden_path, 'images', x)
                    img_labels = Image.open(image_path)
                    # img_labels = Image.open('dataset/broden1_224/images/{}'.format(x))
                    numpy_val = np.array(img_labels)[:, :, 0] + 256* np.array(img_labels)[:, :, 1]
                    code_val = [i for i in np.sort(np.unique(numpy_val))[1:]]
                    labels[full_image_name] += code_val



    images_train, images_valtest = train_test_split(images, test_size=0.4, random_state=42)
    images_val, images_test = train_test_split(images_valtest, test_size=0.5, random_state=42)

    # Sanity checks
    # n_images = len(images)
    assert len(images) == n_images
    assert len(labels) == n_images
    assert len(scene_labels) == n_images

    data = {
            'train': images_train,
            'val': images_val,
            'test': images_test,
            'labels': labels,
            'scene_labels': scene_labels}
    torch.save(data, save_path)
    print("Saved data from {} samples to {}".format(n_images, save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--broden_path', type=str, default='data/broden1_224')
    parser.add_argument('--subset_name', type=str, default='ade20k')
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    # if os.path.exists(args.save_dir):
    #     raise ValueError("Path {} already exists. Aborting".format(args.save_dir))
    ensure_dir(args.save_dir)

    save_broden_subset(
        subset_name=args.subset_name,
        broden_path=args.broden_path,
        save_dir=args.save_dir
    )