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

    for filename in filenames:
        src_path = os.path.join(raw_cifar_dir, filename)
        dst_path = os.path.join(processed_cifar_dir, filename)
        # Not dictionary items
        if 'html' in src_path:
            shutil.copy(src_path,  dst_path)
            print("Saved {} to {}".format(filename, dst_path))
        else:
            src_dict = pickle.load(open(src_path, 'rb'), encoding='bytes')
            dst_dict = {}
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

        