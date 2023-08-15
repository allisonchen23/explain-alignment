import argparse
import numpy as np
import os, sys
from tqdm import tqdm
import torch
import random

sys.path.insert(0, 'src')
# sys.path.insert(0, 'src/ace')
from utils.utils import ensure_dir, read_lists, write_lists, informal_log, load_image, save_image
from ace.ace_helpers import return_superpixels

class ImageSegmenter(object):
    def __init__(self,
             save_dir,
             image_labels_path,
             superpixel_method='slic',
             superpixel_param_dict=None,
             average_image_value=117,
             image_shape=(224, 224)):
        

        self.image_labels_path = image_labels_path
        self.dataset_id = os.path.basename(image_labels_path).split('_imagelabels')[0]

        self.save_dir = os.path.join(save_dir, self.dataset_id)
        ensure_dir(self.save_dir)

        self.log_path = os.path.join(save_dir, 'log.txt')

        self.dictionary_dir = os.path.join(self.save_dir, 'n_patch_dictionaries')
        ensure_dir(self.dictionary_dir)
        self.src_image_paths = None
        self.dst_dirpaths = None

        # Superpixel parameters
        self.superpixel_method = superpixel_method
        if superpixel_param_dict is None:
            # Default ACE parameters
            self.superpixel_param_dict = {
                'n_segments': [15, 50, 80],
                'compactness': [10, 10, 10]
            }
        else:
            self.superpixel_param_dict = superpixel_param_dict
        self.average_image_value = average_image_value
        self.image_shape = image_shape


    def get_image_paths(self,
                        splits=['train', 'val', 'test'],
                        overwrite=False):
        '''
        Make list of all images paths. Save lists in save_dir and return

        Arg(s):
            image_labels_path : str
                .pth file with names of each file in each split (e.g. data/ade20k/full_ade20k_imagelabels.pth)
            save_dir : str
                directory to save .txt file to
            splits : list[str]
                keys in the .pth file to 

        Returns: (list, list)
            list[str] : list of all image paths
            list[str] : list of destination directories
        '''
        # Check directory/file existence
        image_paths_path = os.path.join(self.save_dir, '{}_image_paths.txt'.format(self.dataset_id))
        if os.path.exists(image_paths_path) and not overwrite:
            print("File exists at {}".format(image_paths_path))
            image_paths = read_lists(image_paths_path)
            # return image_paths
        else:
            # Take in the .pth file that has all the files for each split e.g. data/ade20k/full_ade20k_imagelabels.pth
            image_labels = torch.load(self.image_labels_path)
            # Get all paths for train/val/split and append to list
            image_paths = []
            for split in splits:
                split_paths = image_labels[split]
                image_paths += split_paths
            write_lists(image_paths, image_paths_path)
            print("Wrote {} source image paths to {}".format(len(image_paths), image_paths_path))
        # self.src_image_paths = image_paths

        # Create directories to save segmentations in
        segmentation_paths_path = os.path.join(self.save_dir, '{}_ace_segmentation_paths.txt'.format(self.dataset_id))
        image_ids_path = os.path.join(self.save_dir, '{}_image_ids.txt')
        if os.path.exists(segmentation_paths_path) and os.path.exists(image_ids_path) \
            and not overwrite:
            dst_dirs = read_lists(segmentation_paths_path)
        else:
            dst_dirs = []
            image_ids = []
            for src_image_path in image_paths:
                image_id = os.path.basename(src_image_path).split('.jpg')[0]
                segmentation_dir = os.path.join(self.save_dir, 'segmentations', image_id)
                dst_dirs.append(segmentation_dir)
                image_ids.append(image_id)
            write_lists(dst_dirs, segmentation_paths_path)
            print("Wrote {} destination segmentation directory paths to {}".format(len(dst_dirs), segmentation_paths_path))
        # self.dst_dirpaths = dst_dirs
        return image_paths, dst_dirs, image_ids
    
    
    def segment_images(self,
            # image_labels_path,
            # save_dir,
            splits=['train', 'val', 'test'],
            overwrite=False,
            debug=False):
        # segment images

        # Take in src filepaths (call get_image_paths)
        self.src_image_paths, self.dst_dirpaths, self.image_ids = self.get_image_paths(
            splits=splits,
            overwrite=overwrite
        )
        
        # Dictionary for storing # of patches for each image
        # n_patches_dict_path = os.path.join(self.save_dir, 'n_patches_dictionary.pth')
        # if os.path.exists(n_patches_dict_path):
        #     n_patches_dict = torch.load(n_patches_dict_path)
        # else:
        n_patches_dict = {}

        paths = list(zip(self.src_image_paths, self.dst_dirpaths, self.image_ids))

        if debug:
            paths = paths[:100]

        random.shuffle(paths)

        for idx, (src_path, dst_dir, image_id) in enumerate(paths):
            
            # if os.path.isdir(dst_dir) and not overwrite:
            #     continue
            patches_save_dir = os.path.join(dst_dir, 'patches')
            superpixels_save_dir = os.path.join(dst_dir, 'superpixels')

            # If the superpixel and patches directories exist, check that they are complete

            ensure_dir(patches_save_dir)
            ensure_dir(superpixels_save_dir)

            image = load_image(src_path)
            _, image_superpixels, image_patches = return_superpixels(
                index_img=(idx, image),
                method=self.superpixel_method,
                param_dict=self.superpixel_param_dict,
                average_image_value=self.average_image_value,
                image_shape=self.image_shape
            )
            
            # Store image and patch numbers
            n_patches = len(image_superpixels)
            n_patches_dict[dst_dir] = n_patches
            # cur_image_numbers = np.array([idx for i in range(len(image_superpixels))])
            # cur_patch_numbers = np.array([i for i in range(len(image_superpixels))])
            
            # Iterate through all patches/superpixels
            for patch_idx, (superpixel, patch) in enumerate(zip(image_superpixels, image_patches)):
                superpixel_save_path = os.path.join(superpixels_save_dir, 'patch_{}.png'.format(patch_idx))
                patch_save_path = os.path.join(patches_save_dir, 'patch_{}.png'.format(patch_idx))

                save_image(superpixel, superpixel_save_path)
                save_image(patch, patch_save_path)

            print("Saved {} patches in {}".format(
                n_patches, dst_dir
            ))








            # Create directories

            # Call ace._return_superpixels() (actually I think I moved it into ace_helpers.py)
        
        # for src, segmentation_save_dir:
            # if segmentation_save_dir exists:
                # continue 
            # create patch dir and superpixel dir
            # call ace._return_superpixels()
        pass

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='main', help='Run preprocessing or main')
    parser.add_argument('--image_labels_path', type=str, default='data/ade20k/full_ade20k_imagelabels.pth',
                        help='Path to .pth file with list of images in each split')
    parser.add_argument('--save_dir', type=str, default='data/ade20k/ace')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help="Names of splits to segment images from. Default is ['train', 'val', 'test']")
    parser.add_argument('--overwrite', action='store_true', help='Boolean to overwrite data or not')
    args = parser.parse_args()

    # check valid mode
    modes_available = ['preprocess', 'main']
    if args.mode not in modes_available:
        raise ValueError("Mode '{}' not recognized. Please choose from {}".format(args.mode, modes_available))
    
    # Arguments for superpixel
    superpixel_method = 'slic'
    superpixel_param_dict = {
        'n_segments': [15, 50, 80],
        'compactness': [10, 10, 10]
    }
    average_image_value = np.mean([0.485, 0.456, 0.406]) * 255 # ImageNet values
    image_shape = (224, 224) # Shape of ADE20K images

    imseg = ImageSegmenter(
        save_dir=args.save_dir,
        image_labels_path=args.image_labels_path,
        superpixel_method=superpixel_method,
        superpixel_param_dict=superpixel_param_dict,
        average_image_value=average_image_value,
        image_shape=image_shape)
    
    if args.mode == 'preprocess':
        imseg.get_image_paths(
            splits=args.splits,
            overwrite=args.overwrite
        )
    elif args.mode == 'main':
        imseg.segment_images(
            splits=args.splits,
            overwrite=args.overwrite
        )
