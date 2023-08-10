import argparse
import os, sys
from tqdm import tqdm
import torch

sys.path.insert(0, 'src')
from utils.utils import ensure_dir, read_lists, write_lists

def preprocess(image_labels_path, 
               save_dir,
               splits=['train', 'val', 'test'],
               overwrite=False):
    '''
    Make list of all images and save in save_dir and return list of image paths

    Arg(s):
        image_labels_path : str
            .pth file with names of each file in each split (e.g. data/ade20k/full_ade20k_imagelabels.pth)
        save_dir : str
            directory to save .txt file to
        splits : list[str]
            keys in the .pth file to 
    '''
    ensure_dir(save_dir)
    id = os.path.basename(image_labels_path).split('imagelabels')[0]
    save_path = os.path.join(save_dir, '{}_paths.txt'.format(id))
    if os.path.exists(save_path) and not overwrite:
        print("File exists at {}".format(save_path))
        image_paths = read_lists(save_path)
        return image_paths
    # Take in the .pth file that has all the files for each split e.g. data/ade20k/full_ade20k_imagelabels.pth
    image_labels = torch.load(image_labels_path)
    # Get all paths for train/val/split and append to list
    image_paths = []
    for split in splits:
        split_paths = image_labels[split]
        image_paths += split_paths
    write_lists(image_paths, save_path)
    print("Wrote {} paths to {}".format(len(image_paths), save_path))
    return image_paths
    
    
def main():
    # segment images

    # Take in src filepaths
    # Take in save_dir (data/ade20k/ace/segmentations)
    # Create 1 separate lists:
        # segmentation_save_dirs = os.path.join(save_dir, filename like ADE_train_00014363)


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
    args = parser.parse_args()

    # check valid mode
    modes_available = ['preprocess', 'main']
    if args.mode not in modes_available:
        raise ValueError("Mode '{}' not recognized. Please choose from {}".format(args.mode, modes_available))
    
    if args.mode == 'preprocess':
        preprocess(
            image_labels_path=args.image_labels_path,
            save_dir=args.save_dir
        )