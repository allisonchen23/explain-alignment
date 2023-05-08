import os, sys
import torch
# import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, 'src')
from utils.places365_pred_utils import get_class_category_dict
from utils.utils import ensure_dir

# Set variables
places365_dir= os.path.join('data', 'Places365')
index_path = os.path.join(places365_dir, 'places365_val.txt')
image_dir = os.path.join(places365_dir, 'val_256')

train_split = 0.6
seed = 0

data_save_dir = os.path.join('data', 'places365_categories')
data_save_path = os.path.join(data_save_dir, 'places365_imagelabels.pth')

def setup_places365_categories(image_dir,
                               index_path,
                               train_split,
                               data_save_path,
                               seed=0):
    '''
    Saves data in similar format as ade20k/ade20k_imagelabels.pth
    '''
    # Check save paths
    ensure_dir(os.path.dirname(data_save_path))
    if os.path.exists(data_save_path):
        print("Path {} exists. Aborting".format(data_save_path))
        return
    # Check valid train_split
    assert train_split >= 0 and train_split <=1, "train_split of {} invalid".format(train_split)

    # Load dictionary and extract columns
    class_category_dict = get_class_category_dict()

    df = pd.read_csv(index_path, delimiter=' ', header=None)
    df = df.rename(columns={0: 'filename', 1: 'label'})
    filenames = df['filename'].tolist()
    scene_labels = df['label'].tolist()
    scene_category_labels = [class_category_dict[scene_label] for scene_label in scene_labels]

    # Store labels in dictionaries (path to image: label)
    scene_labels_dict = {}
    scene_category_labels_dict = {}
    # Populate dictionaries
    image_paths = []
    for filename, scene_label, scene_category_label in tqdm(zip(filenames, scene_labels, scene_category_labels)):
        image_path = os.path.join(image_dir, filename)
        image_paths.append(image_path)
        assert os.path.exists(image_path), "Path {} does not exist".format(image_path)
        scene_labels_dict[image_path] = scene_label
        scene_category_labels_dict[image_path] = scene_category_label

    # Sanity check
    assert len(filenames) == len(image_paths)
    print("Saved scene and scene category labels")

    # Store in data object
    save_data = {}
    save_data['scene_labels'] = scene_labels_dict
    save_data['scene_category_labels'] = scene_category_labels_dict

    # Store image paths in data frame (needed for shuffling)
    df['image_paths'] = image_paths
    # Randomly split val_train and val_val
    train_df = df.sample(frac=train_split, random_state=seed)
    val_df = df.drop(train_df.index)

    # Assert no overlap between train and val
    assert len(pd.merge(train_df, val_df, how='inner', on=['filename', 'label'])) == 0

    # Store image paths to save data
    save_data['val_train'] = train_df['image_paths'].tolist()
    save_data['val_val'] = val_df['image_paths'].tolist()
    print("Split train/val ({}/{})".format(train_split, 1-train_split))

    # Sanity checks
    for image_path in save_data['val_train']:
        assert image_path in save_data['scene_labels']
        assert image_path in save_data['scene_category_labels']
    for image_path in save_data['val_val']:
        assert image_path in save_data['scene_labels']
        assert image_path in save_data['scene_category_labels']

    torch.save(save_data, data_save_path)
    print("Saved data to {}".format(data_save_path))

if __name__ == "__main__":
    setup_places365_categories(
        image_dir=image_dir,
        index_path=index_path,
        train_split=train_split,
        data_save_path=data_save_path,
        seed=seed
    )
