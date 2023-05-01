import os
import torch

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, 'src')
from utils.utils import load_image, read_lists


class ADE20KDataset(Dataset):
    def __init__(self,
                 path,
                 split,
                 normalize=False,
                 means=None,
                 stds=None,
                 resize=(256, 256),
                 center_crop=(224, 224)):

        data_dictionary = torch.load(path)

        try:
            self.image_paths = data_dictionary[split]
        except:
            raise ValueError("Split '{}' not supported. Try 'train', 'val', or 'test'".format(split))
        self.labels = data_dictionary['labels']

        # Store transforms
        self.transforms = []

        if -1 not in resize:
            self.transforms.append(transforms.Resize(resize, antialias=True))
            print("Resizing to {}".format(resize))
        if -1 not in center_crop:
            self.transforms.append(transforms.CenterCrop(center_crop))
            print("Center cropping to {}".format(center_crop))
        self.transforms.append(transforms.ToTensor())
        if normalize and means is not None and stds is not None:
            self.transforms.append(transforms.Normalize(means, stds))
            print("Normalizing with means of {} and stds of {}".format(means, stds))

        self.transforms = transforms.Compose(self.transforms)
        # Save whether to return paths
        # self.return_paths = return_paths

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Load image
        # image = load_image(image_path, data_format='HWC')
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        # Load label
        label = self.labels[image_path]

        return image #, label

    def __len__(self):
        return len(self.image_paths)


class ImageDataset(Dataset):
    '''
    Dataset for Images given path to list of paths to images and labels

    Arg(s):
        root : str
            path from CWD to root of data storage
        image_paths_path : str
            path to list of relative paths where images are stored
        labels_path : str
            path to file where labels are stored
        return_paths : bool
            whether data loader should return paths or not
        normalize : bool
            whether to normalize or not
        means : list[float]
            mean values for RGB channels
        stds : list[float]
            std values for RGB channels
    '''

    def __init__(self,
                 root,
                 image_paths_path,
                 labels_path,
                 return_paths=True,
                 normalize=False,
                 means=None,
                 stds=None):

        self.root = root
        self.image_paths = read_lists(image_paths_path)
        self.labels = read_lists(labels_path)
        self.n_sample = len(self.image_paths)
        self.return_paths = return_paths

        # Transforms
        self.transforms = [transforms.ToTensor()]
        if normalize:
            assert means is not None and stds is not None
            self.transforms.append(transforms.Normalize(mean=means, std=stds))
        # PyTorch will already switch axes to C x H x W :')
        self.transforms = transforms.Compose(self.transforms)


    def __getitem__(self, index):
        # Obtain path, load image, apply transforms
        image_path = os.path.join(self.root, self.image_paths[index])
        image = load_image(image_path, data_format="HWC")
        image = self.transforms(image)

        # Obtain label
        label = int(self.labels[index])

        # Return data
        if self.return_paths:
            return image, label, image_path
        else:
            return image, label


    def __len__(self):
        return self.n_sample


class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing
    Args:
        root : str
            Root directory of dataset where ``<dataset_type>/*.pt`` will exist.
        dataset_type : str
            Directory in root that containts *.pt
        split : str
            Name of .pt files: training or test
        padding : int
            Amount of edge padding on all sides
        target_transform : (callable, optional)
            A function/transform that takes in the target and transforms it.
    """
    def __init__(self,
                 root: str,
                 dataset_type: str,
                 split: str,
                 padding: int=0,
                 normalize: bool=False,
                 means: list=None,
                 stds: list=None,
                 target_transform=None):
        # Create list of transformations
        transform = []
        if padding > 0:
            transform.append(transforms.Pad(padding, padding_mode='edge'))
        if normalize:
            assert means is not None and stds is not None, "Cannot normalize without means and stds"
            transform.append(transforms.Normalize(mean=means, std=stds))
        if len(transform) > 0:
            transform = transforms.Compose(transform)
        else:
            transform = None

        super(ColoredMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        # Assert valid directory and split
        self.dataset_dir = os.path.join(root, dataset_type)
        assert os.path.isdir(self.dataset_dir), "Directory '{}' does not exist.".format(self.dataset_dir)
        valid_splits = ['training', 'test', 'test_hold_out_50']
        if split not in valid_splits :
            raise ValueError("Data split '{}' not supported. Choose from {}".format(split, valid_splits))

        # Load images and labels
        data_path = os.path.join(self.dataset_dir, "{}.pt".format(split))
        self.data = torch.load(data_path)

        self.images = self.data['images']
        self.labels = self.data['labels']
        self.color_idx = self.data['colors']
        assert len(self.images) == len(self.labels), "Images and labels have different number of samples ({} and {} respectively)".format(
            len(self.images), len(self.labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Obtain image and label
        img = self.images[index]
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)

        target = self.labels[index]

        # Apply transformations (if applicable)
        if self.transform is not None:
            # img = Image.fromarray(np.unit8(img))
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)