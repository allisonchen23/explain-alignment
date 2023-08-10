import os, sys
import torch

sys.path.insert(0, 'src')
from utils.attribute_utils import get_one_hot_attributes, get_frequent_attributes, convert_sparse_to_dense_attributes
from utils.utils import ensure_dir

imagelabels_path = 'data/ade20k/scene_annotated_ade20k_imagelabels.pth'
total_attributes = 1200
save_dir = os.path.join('data', 'ade20k', 'scene_annotated')
def get_frequent_attributes(save_dir,
                            imagelabels_path,
                            n_attributes,
                            total_attributes,
                            overwrite=False):
    ensure_dir(save_dir)
    attributes_save_path = os.path.join(save_dir, 'one_hot_attributes_{}.pth'.format(total_attributes))
    if os.path.exists(attributes_save_path) and not overwrite:
        one_hot_attributes = torch.load(attributes_save_path)
    else: 
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
    
    # Sort attributes in order of frequency
    # save CSV of index, name, and frequency in training
    # ^ load if it already exists
    # Obtain the top n_attr indices
    # Slice those from one_hot_attributes

    

if __name__ == "__main__":
    get_frequent_attributes(
        save_dir=save_dir,
        n_attributes=total_attributes
    )