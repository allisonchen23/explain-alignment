import argparse
import os, sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')
from utils.attribute_utils import get_one_hot_attributes, get_frequent_attributes, sort_attributes, get_attr_name_dict, convert_sparse_to_dense_attributes
from utils.utils import ensure_dir

IMAGELABELS_PATH = 'data/ade20k/scene_annotated_ade20k_imagelabels.pth'
LABELS_CSV_PATH = 'data/broden1_224/label.csv'
total_attributes = 1197
save_dir = os.path.join('data', 'ade20k', 'scene_annotated')

def get_frequent_attributes(save_dir,
                            n_attributes,
                            imagelabels_path=None,
                            labels_path=None,
                            total_attributes=1197,
                            overwrite=False):
    ensure_dir(save_dir)
    attributes_save_path = os.path.join(save_dir, 'one_hot_attributes_{}.pth'.format(total_attributes))
    sorted_attr_save_path = os.path.join(save_dir, 'sorted_attributes.csv')
    # filtered_dense_attributes_save_dir = os.path.join(save_dir, 'dense_{}_attributes'.format(n_attributes))
    # filtered_dense_attributes_save_path = os.path.join(filtered_dense_attributes_save_dir, 'attributes.pth')
    filtered_dense_attributes_save_path = os.path.join(save_dir, 'dense_{}_attributes.pth'.format(n_attributes))
    # filtered_dense_attributes_csv_path = os.path.join(filtered_dense_attributes_save_dir, 'attributes.csv')
    
    # check if dense attributes already exists
    if os.path.exists(filtered_dense_attributes_save_path) and not overwrite:
        dense_one_hot_attributes = torch.load(filtered_dense_attributes_save_path)
        # dense_one_hot_attributes_df = pd.read_csv(filtered_dense_attributes_csv_path)
        print("Loading top {} attribute vectors from {}".format(n_attributes, filtered_dense_attributes_save_path))
        return dense_one_hot_attributes
    
    if os.path.exists(attributes_save_path) and not overwrite:
        one_hot_attributes = torch.load(attributes_save_path)
    else: 
        if imagelabels_path is None:
            imagelabels_path = IMAGELABELS_PATH
        
        imagelabels = torch.load(imagelabels_path)
        paths = {
            'train': imagelabels['train'],
            'val': imagelabels['val'],
            'test': imagelabels['test']
        }
        one_hot_attributes = get_one_hot_attributes(
            data=imagelabels,
            paths=paths,
            n_attr=total_attributes
        )
    
    # Load sorted attribute file or save CSV of index, name, and frequency of attributes in training
    if os.path.exists(sorted_attr_save_path) and not overwrite:
        sorted_attr_df = pd.read_csv(sorted_attr_save_path)
        sort_attr_idxs = sorted_attr_df['attribute_idxs'].to_numpy()
    else: 
        if labels_path is None:
            labels_path = LABELS_CSV_PATH
        # Load attribute names
        attr_name_dictionary = get_attr_name_dict(labels_csv_path=labels_path)
        
        # Sort attributes in order of frequency
        sort_attr_idxs, sort_attr_occurrences = sort_attributes(one_hot_attributes)
        print("Most frequent attributes: ")
        for i in range(5):
            print("{} ({}): {} occurrences".format(
                sort_attr_idxs[i],
                attr_name_dictionary[sort_attr_idxs[i]],
                sort_attr_occurrences[i]
            ))
        sort_attr_names = []
        for attr_idx in sort_attr_idxs:
            sort_attr_names.append(attr_name_dictionary[attr_idx])
        sorted_attr_df = pd.DataFrame({
            'attribute_idxs': sort_attr_idxs,
            'occurrences': sort_attr_occurrences,
            'names': sort_attr_names
        })
        sorted_attr_df.to_csv(sorted_attr_save_path)
    
    if n_attributes <=0 or n_attributes > len(sort_attr_idxs):
        print("Value for n_attributes ({}) too large. Setting to {} (number of non-zero attributes)".format(
            n_attributes, len(sort_attr_idxs)
        ))
        n_attributes = len(sort_attr_idxs)
    
    # Obtain the top n_attr indices
    top_attr_idxs = sort_attr_idxs[:n_attributes]
    
    # Slice those from one_hot_attributes
    dense_one_hot_attributes = {}
    for split in one_hot_attributes.keys():
        dense_one_hot_attributes[split] = one_hot_attributes[split][:, top_attr_idxs]
    torch.save(dense_one_hot_attributes, filtered_dense_attributes_save_path)
    print("Saved dense top {} attribute vectors from {}".format(n_attributes, filtered_dense_attributes_save_path))
    return dense_one_hot_attributes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_attributes', type=int, default=150, help='Number of attributes to save based on frequency in training')

    args = parser.parse_args()

    get_frequent_attributes(
        save_dir=save_dir,
        imagelabels_path=imagelabels_path,
        labels_path=labels_csv_path,
        n_attributes=args.n_attributes,
        total_attributes=total_attributes
    )