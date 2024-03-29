import os
import torch

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
import sys
import pickle
import numpy as np
from PIL import Image

sys.path.insert(0, 'src')
from utils.utils import load_image, read_lists

class KDDataset(Dataset):
    def __init__(self,
                 input_features_path,
                 labels_path,
                 split,
                 out_type,
                 dtype='float32'):
        available_splits = ['train', 'val', 'test']
        available_out_types = ['outputs', 'probabilities', 'predictions']
        assert split in available_splits, "Received invalid split '{}'. Must be one of {}".format(split, available_splits)
        assert out_type in available_out_types, \
            "Received invalid out_type '{}'. Must be one of {}".format(out_type, available_out_types)

        # Save in features
        input_features = torch.load(input_features_path)
        self.input_features = input_features[split].astype(dtype)
        if not torch.is_tensor(self.input_features):
            self.input_features = torch.from_numpy(self.input_features)

        # Save labels
        labels = torch.load(labels_path)
        self.labels = labels[split][out_type]
        # If soft labels, make sure appropriate data type
        if len(self.labels.shape) == 2:
            self.labels = self.labels.astype(dtype)

        if not torch.is_tensor(self.labels):
            self.labels = torch.from_numpy(self.labels)
        # Metadata
        self.n_samples = len(self.labels)
        assert len(self.input_features) == self.n_samples, \
            "Received unequal lengths for input features ({}) and labels ({})".format(len(self.input_features), self.n_samples)
        self.input_features_path = input_features_path
        self.labels_path = labels_path
        self.dtype = dtype

    def __getitem__(self, index):
        features = self.input_features[index] #.to(self.dtype)
        label = self.labels[index] #.to(self.dtype)

        return features, label

    def __len__(self):
        return self.n_samples

class CIFAR10TorchDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 to_tensor=True,
                 normalize=True,
                 means=[0.4914, 0.4822, 0.4465],
                 stds=[0.2471, 0.2435, 0.2616]):
        
        images = []
        labels = []
        assert split in ['train', 'test'], "Invalid split '{}'. Must be 'train' or 'test'".format(split)
        if split == 'train':
            files = ['data_batch_{}'.format(i) for i in range(1,6)]
        else:
            files = ['test_batch']
            
        for file in files:
            path = os.path.join(dataset_dir, file)
            data = pickle.load(open(path, 'rb'))
            cur_images = data['data']
            cur_images = np.reshape(cur_images, (-1, 3, 32, 32))
            images.append(cur_images)
            
            cur_labels = np.array(data['labels'])
            labels.append(cur_labels)
        
        self.images = np.concatenate(images, axis=0)
        self.images = np.transpose(self.images, (0, 2, 3, 1))
        self.labels = np.concatenate(labels, axis=0)
        self.n_samples = len(self.labels)
        self.dataset_dir = dataset_dir
                
        # Create transformations
        self.transforms = [transforms.ToTensor()]  # changes dims H x W x C -> C x H x W and scales to [0, 1]
        if normalize:
            self.transforms.append(transforms.Normalize(means, stds))
        self.transforms = transforms.Compose(self.transforms)
    
    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return self.n_samples
    
class ImageDataset(Dataset):
    def __init__(self,
                 path,
                 split,
                 return_label=False,
                 label_key=None,
                 normalize=False,
                 means=None,
                 stds=None,
                 resize=(256, 256),
                 center_crop=(224, 224)):

        data_dictionary = torch.load(path)

        # Obtain image paths
        try:
            self.image_paths = data_dictionary[split]
        except:
            raise ValueError("Split '{}' not supported. Try 'train', 'val', or 'test'".format(split))

        # Obtain labels
        self.return_label = return_label
        if self.return_label:
            try:
                self.labels = data_dictionary[label_key]
            except:
                raise ValueError("Label key '{}' not found in data dictionary. Try one of {}".format(
                    label_key, list(data_dictionary.keys())
                ))


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


        if not self.return_label:
            return image
        else:
            # Load label
            label = self.labels[image_path]
            return image, label

    def __len__(self):
        return len(self.image_paths)


# class ImageDataset(Dataset):
#     '''
#     Dataset for Images given path to list of paths to images and labels

#     Arg(s):
#         root : str
#             path from CWD to root of data storage
#         image_paths_path : str
#             path to list of relative paths where images are stored
#         labels_path : str
#             path to file where labels are stored
#         return_paths : bool
#             whether data loader should return paths or not
#         normalize : bool
#             whether to normalize or not
#         means : list[float]
#             mean values for RGB channels
#         stds : list[float]
#             std values for RGB channels
#     '''

#     def __init__(self,
#                  root,
#                  image_paths_path,
#                  labels_path,
#                  return_paths=True,
#                  normalize=False,
#                  means=None,
#                  stds=None):

#         self.root = root
#         self.image_paths = read_lists(image_paths_path)
#         self.labels = read_lists(labels_path)
#         self.n_sample = len(self.image_paths)
#         self.return_paths = return_paths

#         # Transforms
#         self.transforms = [transforms.ToTensor()]
#         if normalize:
#             assert means is not None and stds is not None
#             self.transforms.append(transforms.Normalize(mean=means, std=stds))
#         # PyTorch will already switch axes to C x H x W :')
#         self.transforms = transforms.Compose(self.transforms)


#     def __getitem__(self, index):
#         # Obtain path, load image, apply transforms
#         image_path = os.path.join(self.root, self.image_paths[index])
#         image = load_image(image_path, data_format="HWC")
#         image = self.transforms(image)

#         # Obtain label
#         label = int(self.labels[index])

#         # Return data
#         if self.return_paths:
#             return image, label, image_path
#         else:
#             return image, label


#     def __len__(self):
#         return self.n_sample


# class ColoredMNIST(datasets.VisionDataset):
#     """
#     Colored MNIST dataset for testing
#     Args:
#         root : str
#             Root directory of dataset where ``<dataset_type>/*.pt`` will exist.
#         dataset_type : str
#             Directory in root that containts *.pt
#         split : str
#             Name of .pt files: training or test
#         padding : int
#             Amount of edge padding on all sides
#         target_transform : (callable, optional)
#             A function/transform that takes in the target and transforms it.
#     """
#     def __init__(self,
#                  root: str,
#                  dataset_type: str,
#                  split: str,
#                  padding: int=0,
#                  normalize: bool=False,
#                  means: list=None,
#                  stds: list=None,
#                  target_transform=None):
#         # Create list of transformations
#         transform = []
#         if padding > 0:
#             transform.append(transforms.Pad(padding, padding_mode='edge'))
#         if normalize:
#             assert means is not None and stds is not None, "Cannot normalize without means and stds"
#             transform.append(transforms.Normalize(mean=means, std=stds))
#         if len(transform) > 0:
#             transform = transforms.Compose(transform)
#         else:
#             transform = None

#         super(ColoredMNIST, self).__init__(root, transform=transform,
#                                     target_transform=target_transform)

#         # Assert valid directory and split
#         self.dataset_dir = os.path.join(root, dataset_type)
#         assert os.path.isdir(self.dataset_dir), "Directory '{}' does not exist.".format(self.dataset_dir)
#         valid_splits = ['training', 'test', 'test_hold_out_50']
#         if split not in valid_splits :
#             raise ValueError("Data split '{}' not supported. Choose from {}".format(split, valid_splits))

#         # Load images and labels
#         data_path = os.path.join(self.dataset_dir, "{}.pt".format(split))
#         self.data = torch.load(data_path)

#         self.images = self.data['images']
#         self.labels = self.data['labels']
#         self.color_idx = self.data['colors']
#         assert len(self.images) == len(self.labels), "Images and labels have different number of samples ({} and {} respectively)".format(
#             len(self.images), len(self.labels))

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         # Obtain image and label
#         img = self.images[index]
#         if not torch.is_tensor(img):
#             img = torch.from_numpy(img)

#         target = self.labels[index]

#         # Apply transformations (if applicable)
#         if self.transform is not None:
#             # img = Image.fromarray(np.unit8(img))
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.images)