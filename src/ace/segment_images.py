import argparse
from datetime import datetime
import numpy as np
import os, sys
import shutil
import time
from tqdm import tqdm
import torch
import random

sys.path.insert(0, 'src')
# sys.path.insert(0, 'src/ace')
from utils.utils import ensure_dir, read_lists, write_lists, informal_log, load_image, save_image
from ace.ace_helpers import return_superpixels, load_features_model, load_images_from_files, get_features

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
        
    
        self.dictionary_dir = os.path.join(self.save_dir, 'n_patch_dictionaries')
        self.features_dir = os.path.join(self.save_dir, 'superpixel_features')
        self.features_paths_dir = os.path.join(self.save_dir, 'features_paths')
        self.log_dir = os.path.join(self.save_dir, 'logs')
        self.log_path = os.path.join(self.log_dir, 'log.txt')
        
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

    def preprocess(self,
                   splits=['train', 'val', 'test'],
                   overwrite=False):
        
        paths = [
            self.save_dir,
            self.dictionary_dir,
            self.features_dir,
            self.features_paths_dir,
            self.log_dir
        ]
        # Ensure directories by overwriting
        if overwrite:
            for path in paths:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                ensure_dir(path)
            # if os.path.isdir(self.save_dir):
            #     shutil.rmtree(self.save_dir)
            # ensure_dir(self.save_dir)
            # if os.path.isdir(self.dictionary_dir):
            #     shutil.rmtree(self.dictionary_dir)
            # ensure_dir(self.dictionary_dir)

            # if os.path.isdir(self.features_dir):
            #     shutil.rmtree(self.features_dir)
            # ensure_dir(self.features_dir)
        # Don't overwrite, just ensure they exist
        else:
            for path in paths:
                ensure_dir(path)
            # ensure_dir(self.save_dir)
            # ensure_dir(self.dictionary_dir) 
            # ensure_dir(self.features_dir)

        print("Created directories:")
        for path in paths:
            print(path)
        self.get_image_paths(
            splits=splits,
            overwrite=overwrite)
        
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
        image_ids_path = os.path.join(self.save_dir, '{}_image_ids.txt'.format(self.dataset_id))
        if os.path.exists(segmentation_paths_path) and os.path.exists(image_ids_path) \
            and not overwrite:
            dst_dirs = read_lists(segmentation_paths_path)
            image_ids = read_lists(image_ids_path)
            print("File exists at {}".format(segmentation_paths_path))
            print("File exists at {}".format(image_ids_path))
        else:
            dst_dirs = []
            image_ids = []
            for src_image_path in image_paths:
                image_id = os.path.basename(src_image_path).split('.jpg')[0]
                segmentation_dir = os.path.join(self.save_dir, 'segmentations', image_id)
                dst_dirs.append(segmentation_dir)
                image_ids.append(image_id)
            write_lists(dst_dirs, segmentation_paths_path)
            write_lists(image_ids, image_ids_path)
            print("Wrote {} destination segmentation directory paths to {}".format(len(dst_dirs), segmentation_paths_path))
            print("Wrote {} image ids paths to {}".format(len(image_ids), image_ids_path))
        # self.dst_dirpaths = dst_dirs
        return image_paths, dst_dirs, image_ids
    
    
    def segment_images(self,
            splits=['train', 'val', 'test'],
            overwrite=False,
            debug=False):
        '''
        Function that can be called in parallel to segment images and save patches and superpixels.
        Saves number of patches per image in a dictionary in self.save_dir/self.dataset_id/n_patch_dictionaries
        '''
        # Take in src filepaths (call get_image_paths)
        self.src_image_paths, self.dst_dirpaths, self.image_ids = self.get_image_paths(
            splits=splits,
            overwrite=overwrite
        )
        
        # Dictionary for storing # of patches for each image
        n_patches_dict = {}

        paths = list(zip(self.src_image_paths, self.dst_dirpaths, self.image_ids))

        if debug:
            paths = paths[:10]

        random.shuffle(paths)

        for idx, (src_path, dst_dir, image_id) in enumerate(paths):
            
            if os.path.isdir(dst_dir) and not overwrite:
                continue
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

            informal_log("{},{}".format(dst_dir, n_patches), self.log_path, timestamp=False)

        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        n_patches_dictionary_path = os.path.join(self.dictionary_dir, '{}.pth'.format(timestamp))
        time.sleep(random.random())
        while os.path.exists(n_patches_dictionary_path):
            timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
            n_patches_dictionary_path = os.path.join(self.dictionary_dir, '{}.pth'.format(timestamp))
            time.sleep(random.random())

        torch.save(n_patches_dict, n_patches_dictionary_path)

    def save_features(self,
                      splits,
                      model_arch,
                      n_classes,
                      device,
                      model_checkpoint_path,
                      batch_size,
                      channel_mean,
                      overwrite=False,
                      debug=False):
        log_path = os.path.join(self.log_dir, 'features_log.txt')

        # Store paths to directories
        self.src_image_paths, self.dst_dirpaths, self.image_ids = self.get_image_paths(
            splits=splits,
            overwrite=False
        )
        features_save_paths = [
            os.path.join(self.features_dir, '{}_features.pth'.format(image_id)) \
                for image_id in self.image_ids
        ]

        # Load model
        _, feature_model = load_features_model(
            arch=model_arch,
            n_classes=n_classes,
            device=device,
            checkpoint_path=model_checkpoint_path
        )

        paths = list(zip(self.dst_dirpaths, features_save_paths, self.image_ids))
        if debug:
            paths = paths[:10]

        random.shuffle(paths)

        # features = {}
        stored_features_paths = []
        for image_idx, (dst_dir, features_path, image_id) in enumerate(paths):
            if os.path.exists(features_path) and not overwrite:
                continue

            superpixel_dir = os.path.join(dst_dir, 'superpixels')
            superpixel_paths = [os.path.join(superpixel_dir, patch_name) for patch_name in os.listdir(superpixel_dir)]
            patch_ids = [int(os.path.basename(superpixel_path).split('patch_')[1].split('.png')[0]) \
                for superpixel_path in superpixel_paths]
            sort_idxs = np.argsort(patch_ids)
            patch_ids = np.array(patch_ids)[sort_idxs]
            superpixel_paths = np.array(superpixel_paths)[sort_idxs]
            
            # superpixel_patches : np.array[N x H x W x C]
            superpixel_patches = load_images_from_files(
                filenames=superpixel_paths,
                max_imgs=len(superpixel_paths),
                return_filenames=False,
                do_shuffle=False,
                run_parallel=False,
                shape=self.image_shape
            )

            superpixel_features = get_features(
                dataset=superpixel_patches,
                features_model=feature_model,
                device=device,
                batch_size=batch_size,
                channel_mean=channel_mean
            )
            # Save features
            torch.save(superpixel_features, features_path)
            stored_features_paths.append(features_path)

            informal_log("{}".format(features_path), log_path, timestamp=False)

        # Save paths that we stored features to
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        stored_features_paths_path = os.path.join(self.features_paths_dir, 'stored_features_paths_{}.txt'.format(timestamp))
        time.sleep(random.random())
        while os.path.exists(stored_features_paths_path):
            timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
            stored_features_paths_path = os.path.join(self.features_paths_dir, 'stored_features_paths_{}.txt'.format(timestamp))
            time.sleep(random.random())
        write_lists(stored_features_paths, stored_features_paths_path)


    def verify_superpixels(self, remove_corrupt=True):
        verify_log_path = os.path.join(self.log_dir, 'verification_log.txt')
        dictionary_paths = [os.path.join(self.dictionary_dir, dict_name) for dict_name in sorted(os.listdir(self.dictionary_dir))]
        master_dictionary = {}

        # Combine dictionaries into 1 master dictionary
        for dictionary_path in dictionary_paths:
            try:
                n_patches_dict = torch.load(dictionary_path)
                master_dictionary.update(n_patches_dict)
            except: 
                raise ValueError("Dictionary at {} corrupted.".format(dictionary_path))
        
        # Check number of paths in dictionary is equal to number of directories in 'segmentations'
        segmentation_save_dir = os.path.join(self.save_dir, 'segmentations')
        segmentation_dirs = [os.path.join(segmentation_save_dir, image_id) \
                              for image_id in os.listdir(segmentation_save_dir)]
        # assert len(segmentation_dirs) == len(master_dictionary)
        
        # For each segmentation directory, check number of patches is as expected
        for segmentation_dir in tqdm(segmentation_dirs):
            if segmentation_dir not in master_dictionary:
                informal_log("{},Not found in dictionary".format(segmentation_dir))
                if remove_corrupt:
                    shutil.rmtree(segmentation_dir)
                continue
            n_patches = master_dictionary[segmentation_dir]
            superpixel_dir = os.path.join(segmentation_dir, 'superpixels')
            patches_dir = os.path.join(segmentation_dir, 'patches')

            # Check if correct number of patches for superpixels
            if len(os.listdir(superpixel_dir)) != n_patches:
                informal_log("{},S,expected:{},found:{}".format(
                    segmentation_dir,n_patches,len(os.listdir(superpixel_dir))
                ), verify_log_path, timestamp=False)
                if remove_corrupt:
                    shutil.rmtree(segmentation_dir)
                    master_dictionary.pop(segmentation_dir)
            # Check if correct number of patches for patches    
            if len(os.listdir(patches_dir)) != n_patches:
                informal_log("{},P,expected:{},found:{}".format(
                    segmentation_dir,n_patches,len(os.listdir(patches_dir))
                ), verify_log_path, timestamp=False)
                if remove_corrupt:
                    shutil.rmtree(segmentation_dir)
                    master_dictionary.pop(segmentation_dir)

        # Assert cleanliness
        # Get number of image folders in 'segmentations' now
        segmentation_dirs = [os.path.join(segmentation_save_dir, image_id) \
                              for image_id in os.listdir(segmentation_save_dir)]
        assert len(segmentation_dirs) == len(master_dictionary)
        for segmentation_dir in segmentation_dirs:
            assert segmentation_dir in master_dictionary
        # Clear out the n_patches_dictionary directory and store validated dictionary
        # for dictionary_path in dictionary_paths:
        #     os.remove(dictionary_path)

        master_dictionary_path = os.path.join(
            os.path.dirname(self.dictionary_dir), 
            'validate', 'validated_n_patches_dictionary.pth')
        ensure_dir(os.path.dirname(master_dictionary_path))
        torch.save(master_dictionary,master_dictionary_path)
        informal_log("Saved updated dictionary of n_patches to {}".format(master_dictionary_path))

        return

    def _consolidate_features_by_split(self, features_dict, overwrite=False):
        splits = ['train', 'test', 'val']
        
        # Obtain paths to features that belong in each split
        image_labels = torch.load(self.image_labels_path)
        split_features_save_path_template = os.path.join(self.save_dir, '{}_superpixel_features.pth')
        for split in splits:
            split_features_save_path = split_features_save_path_template.format(split)
            if os.path.exists(split_features_save_path) and not overwrite:
                continue
            split_image_paths = image_labels[split]
            split_features_paths = []
            split_features = []
            for image_path in split_image_paths:
                image_id = os.path.basename(image_path).split('.')[0]
                features_path = os.path.join(self.features_dir, '{}_features.pth'.format(image_id))
                split_features_paths.append(features_path)

                split_features.append(features_dict[features_path])
            torch.save(split_features, split_features_save_path)
            # features_paths[split] = split_features_paths
        
    def verify_features(self,
                        consolidate=True):
        '''
        Verify that the number of features for each image corresponds with the number of patches
        in the dictionary at <self.dictionary_dir>/validate/validated_n_patches_dictionary.pth
        '''
        master_dictionary_path = os.path.join(
            os.path.dirname(self.dictionary_dir), 
            'validate', 'validated_n_patches_dictionary.pth')
        if not os.path.exists(master_dictionary_path):
            print("Run with mode = verify_superpixels first.")
            return
        master_dictionary = torch.load(master_dictionary_path)
        updated_master_dictionary = {}
        segmentation_dirs = master_dictionary.keys()
        # convert master dictionary from segmentation_dirs to image_ids as keys
        for segmentation_dir, n_patches in master_dictionary.items():
            image_id = os.path.basename(segmentation_dir)
            updated_master_dictionary[image_id] = n_patches
            
        features_files = os.listdir(self.features_dir)
        features_paths = [os.path.join(self.features_dir, filename) for filename in features_files]
        # Ensure the features at features_path has the correct number of features (1 per patch)
        features_dict = {}
        for features_path in features_paths:
            features = torch.load(features_path)
            n_features = features.shape[0]
            image_id = os.path.basename(features_path).split('_features.pth')[0]

            if n_features != updated_master_dictionary[image_id]:
                print("{} has {} patches in the dictionary but {} features".format(
                    image_id, updated_master_dictionary[image_id], n_features))
            features_dict[features_path] = features
        
        if consolidate:
            self._consolidate_features_by_split(
                features_dict=features_dict
            )

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='main', help='Run preprocessing or main')
    parser.add_argument('--image_labels_path', type=str, default='data/ade20k/full_ade20k_imagelabels.pth',
                        help='Path to .pth file with list of images in each split')
    parser.add_argument('--save_dir', type=str, default='data/ade20k/ace')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help="Names of splits to segment images from. Default is ['train', 'val', 'test']")
    parser.add_argument('--overwrite', action='store_true', help='Boolean to overwrite data or not')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # check valid mode
    modes_available = ['preprocess', 'main', 'verify_superpixels', 'features', 'verify_features']
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

    # Args for features model. I am lazy
    model_checkpoint_path = os.path.join('checkpoints/resnet18_places365.pth')
    model_arch = 'resnet18'
    n_classes = 365
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    channel_mean = True

    imseg = ImageSegmenter(
        save_dir=args.save_dir,
        image_labels_path=args.image_labels_path,
        superpixel_method=superpixel_method,
        superpixel_param_dict=superpixel_param_dict,
        average_image_value=average_image_value,
        image_shape=image_shape)
    
    if args.mode == 'preprocess':
        imseg.preprocess(
            splits=args.splits,
            overwrite=args.overwrite
        )
    elif args.mode == 'main':
        imseg.segment_images(
            splits=args.splits,
            overwrite=args.overwrite,
            debug=args.debug
        )
    elif args.mode == 'verify_superpixels':
        imseg.verify_superpixels()
    elif args.mode == 'features':
        imseg.save_features(
            splits=args.splits,
            model_arch=model_arch,
            n_classes=n_classes,
            device=device,
            model_checkpoint_path=model_checkpoint_path,
            batch_size=batch_size,
            channel_mean=channel_mean,
            debug=args.debug
        )
    elif args.mode == 'verify_features':
        imseg.verify_features(
            consolidate=True
        )

