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

BRODEN_SUBSETS = ['ade20k', 'dtd', 'pascal']


def save_broden_subset(subset_name: str,
                       broden_path: str,
                       save_dir: str):

    # Check valid subset name
    assert subset_name in BRODEN_SUBSETS, "Subset '{}' not found in available subsets: {}".format(subset_name, BRODEN_SUBSETS)
    print("Saving {} subset of Broden dataset from '{}' to '{}'".format(
        subset_name,
        broden_path,
        save_dir
    ))

    index_path = os.path.join(broden_path, 'index.csv')
    image_df = pd.read_csv(index_path)

    # Define variables
    images = []
    labels = {}

    # Iterate through all images in index.csv
    for idx in tqdm(image_df.index):
        if (image_df['image'][idx]).split('/')[0] != subset_name:
            continue
        full_image_name = os.path.join(broden_path, 'images/{}'.format(image_df['image'][idx]))
        #print(idx)
        images.append(full_image_name)
        labels[full_image_name] = []

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

    save_path = os.path.join(save_dir, '{}_imagelabels.pth'.format(subset_name))
    data = {
            'train': images_train,
            'val':images_val,
            'test':images_test,
            'labels':labels}
    torch.save(data, save_path)
    # with open(save_path, 'wb+') as handle:
    #     pickle.dump({
    #         'train': images_train,
    #         'val':images_val,
    #         'test':images_test,
    #         'labels':labels}, handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--broden_path', type=str, default='data/broden1_224')
    parser.add_argument('--subset_name', type=str, default='ade20k')
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    if os.path.exists(args.save_dir):
        raise ValueError("Path {} already exists. Aborting".format(args.save_dir))
    ensure_dir(args.save_dir)

    save_broden_subset(
        subset_name=args.subset_name,
        broden_path=args.broden_path,
        save_dir=args.save_dir
    )