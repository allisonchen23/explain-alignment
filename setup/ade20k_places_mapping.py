import pandas as pd
import pickle
import os, sys
import torch

sys.path.insert(0, 'src')
from utils.utils import read_lists, list_to_dict, ensure_dir

def get_class_mapping(overwrite=False):
    '''
    Given a csv from my Google Sheets annotations, create 2 dictionaries:
        1) Mapping between ADE20K class indices -> Places 365 Scene Category indices
        2) Mapping between ADE20K class names -> Places 365 Scene Category names

    Load if it already exists and not overwriting, otherwise write to disk
    '''
    map_dir = os.path.join('data', 'ade20k', 'ade_places_category_mapping')
    idx_map_path = os.path.join(map_dir, 'idx_map.pkl')
    name_map_path = os.path.join(map_dir, 'name_map.pkl')
    if os.path.exists(idx_map_path) and os.path.exists(name_map_path) and not overwrite:
        with open(idx_map_path, 'rb') as file:
            idx_map = pickle.load(file)
        with open(name_map_path, 'rb') as file:
            name_map = pickle.load(file)
        return idx_map, name_map
    
    csv_path = os.path.join('data', 'ade20k', 'ade20k_places365_scene_category_mapping.csv')
    scene_categories_path = 'data/ade20k/scene_categories.txt'
    scene_categories_dict = list_to_dict(read_lists(scene_categories_path))
    df = pd.read_csv(csv_path)

    idx_map = {}
    name_map = {}

    for row_idx, row in df.iterrows():
        if row_idx == 9:
            print(row)
        ade_class = row['name']
        ade_class_idx = row['number']
        places_category = row['Scene Category']

        # Check for empty values that I was unsure about
        if pd.isna(places_category):
            places_category = None
            places_category_idx = -1
        else:
            places_category_idx = scene_categories_dict[places_category]

        idx_map[ade_class_idx] = places_category_idx
        name_map[ade_class] = places_category
    
    ensure_dir(map_dir)
    with open(idx_map_path, 'wb') as file:
        pickle.dump(idx_map, file)
    with open(name_map_path, 'wb') as file:
        pickle.dump(name_map, file)
    return idx_map, name_map

def get_places_labels(overwrite=False):
    image_labels_path = 'data/ade20k/full_ade20k_imagelabels.pth'
    places_category_list_path = 'data/places365_categories/scene_categories.txt'
    ade20k_places_idx_map_path = 'data/ade20k/ade_places_category_mapping/idx_map.pkl'
    
    image_labels = torch.load(image_labels_path)
    
    # Destination Paths
    save_path = os.path.join(
        os.path.dirname(ade20k_places_idx_map_path), 
        'full_ade20k_image_labels_scenes.pth')
    if os.path.exists(save_path) and not overwrite:
        print("Places categories already in {}".format(image_labels_path))
        with open(save_path, 'rb') as f:
            return pickle.load(f)
        
    places_category_list = read_lists(places_category_list_path)
    with open(ade20k_places_idx_map_path, 'rb') as f:
        ade20k_places_idx_map = pickle.load(f)
    
    ade20k_scene_annotations = image_labels['scene_labels']
    places_scene_annotations = {}
    n_images_per_scene_counter = {}

    for path, scene_idx in ade20k_scene_annotations.items():
        try:
            places_idx = ade20k_places_idx_map[scene_idx]
        except:
            # If no scene label, set to -1
            places_idx = -1
        places_scene_annotations[path] = places_idx
        if places_idx in n_images_per_scene_counter:
            n_images_per_scene_counter[places_idx] += 1
        else:
            n_images_per_scene_counter[places_idx] = 1
    image_labels['places_labels'] = places_scene_annotations
    
    # Print counter
    for scene_category_idx, category_name in enumerate(places_category_list):
        count = n_images_per_scene_counter[scene_category_idx]
        print("{} ({}): {} images".format(category_name, scene_category_idx, count))
    # Save file
    torch.save(image_labels, save_path)
    return image_labels
    
    
if __name__ == '__main__':
    # get_class_mapping()
    get_places_labels()