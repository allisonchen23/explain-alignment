import torch
import numpy as np
import os, sys
import shutil
import pickle

sys.path.insert(0, 'src')
from utils.utils import ensure_dir

def setup_cifar10(raw_cifar_dir, processed_cifar_dir):
    ensure_dir(processed_cifar_dir)

    raw_file_paths = [os.path.join(raw_cifar_dir, filename) for filename in os.listdir(raw_cifar_dir)]
    filenames = os.listdir(raw_cifar_dir)
    
    print("Processing bytes into strings...")
    for filename in filenames:
        src_path = os.path.join(raw_cifar_dir, filename)
        dst_path = os.path.join(processed_cifar_dir, filename)
        if not os.path.exists(dst_path):
            # Not dictionary files
            if 'html' in src_path:
                shutil.copy(src_path,  dst_path)
                print("Saved {} to {}".format(filename, dst_path))
            else:
                src_dict = pickle.load(open(src_path, 'rb'), encoding='bytes')
                dst_dict = {}
                # Copy each dictionary over converting bytes -> strings
                for key, value in src_dict.items():
                    decoded_key = key.decode("utf-8")
                    # Decode bytes
                    if type(value) == bytes:
                        decoded_value = value.decode("utf-8")
                    # Decode list of bytes
                    elif type(value) == list and type(value[0]) == bytes: 
                        decoded_value = list(map(lambda x: x.decode("utf-8"), value))
                    else:
                        decoded_value = value 
                    dst_dict[decoded_key] = decoded_value
                pickle.dump(dst_dict, open(dst_path, 'wb'))
                print("Saved {} to {}".format(filename, dst_path))
        else:
            print("File already exists at {}".format(dst_path))
                
# Additionally, save labels as a dictionary split into train and test
def consolidate(processed_cifar_dir):
    save_path = os.path.join(processed_cifar_dir, 'cifar10_image_labels.pth')
    if os.path.exists(save_path):
        print("{} already exists. Aborting".format(save_path))
        return
    
    train_filenames = ['data_batch_{}'.format(i) for i in range(1,6)]
    test_filenames = ['test_batch']
    
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    save_data = {
        'train': {
            'images': [],
            'predictions': []
        },
        'test': {
            'images': [],
            'predictions': []
        }
    }
    for split, filenames in zip(['train', 'test'], [train_filenames, test_filenames]):
        print("Processing {} split".format(split))
        filepaths = [os.path.join(processed_cifar_dir, filename) for filename in filenames]
        for filepath in filepaths:
            cur_data = pickle.load(open(filepath, 'rb'))
            cur_images = cur_data['data']
            cur_images = np.reshape(cur_images, (-1, 3, 32, 32))
            cur_images = np.transpose(cur_images, (0, 2, 3, 1))
            
            cur_labels = np.array(cur_data['labels'])
            save_data[split]['images'].append(cur_images)
            save_data[split]['predictions'].append(cur_labels)
            
        # Concatenate lists
        save_data[split]['images'] = np.concatenate(save_data[split]['images'], axis=0)
        save_data[split]['predictions'] = np.concatenate(save_data[split]['predictions'], axis=0)
                
    torch.save(save_data, save_path)
    print("Saved arrays to {}".format(save_path))
            
if __name__ == "__main__":
    raw_cifar_dir = 'data/cifar-10-batches-py'
    processed_cifar_dir = 'data/cifar10-processed'
    # setup_cifar10(
    #     raw_cifar_dir=raw_cifar_dir,
    #     processed_cifar_dir=processed_cifar_dir)
    
    consolidate(processed_cifar_dir=processed_cifar_dir)
        