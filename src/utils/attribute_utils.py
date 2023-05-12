import numpy as np
import os, sys
from tqdm import tqdm

sys.path.insert(0, 'src')


def get_one_hot_attributes(data, paths, n_attr, splits=['train', 'val', 'test']):
    attributes = {}
    # for split in splits:
    #     attributes[split] = []

    for split in splits:
        split_paths = paths[split]
        print("Processing attributes for {} split".format(split))
        for path in tqdm(split_paths):
            # Obtain attributes and covnvert to one hot
            cur_attributes = data['labels'][path]
            one_hot_attributes = np.zeros(n_attr)
            one_hot_attributes[cur_attributes] = 1
            attributes[split].append(one_hot_attributes)
        attributes[split] = np.stack(attributes[split], axis=0)
    return attributes

def get_frequent_attributes(attributes,
                               frequency_threshold=150,
                               splits=['train', 'val', 'test']):
    '''
    Given dictionary of 1-hot encoded attributes, return dictionary of same format with only frequent attributes

    Arg(s):
        attributes : dict[str : np.array]
            keys: split ['train', 'val', 'test']
            values: one-hot encoded attributes
        frequency_threshold : int
            number of occurrences in training for an attribute to be considered 'frequent'
        splits : list[str]
            list of split names to key dictionaries

    Returns:
        freq_attributes_dict : dict[str : np.array]
    '''
    train_counts = np.sum(attributes['train'], axis=0)

    # Obtain one-hot encoding of attributes that exceed frequency threshold
    freq_attributes_one_hot = np.where(train_counts > frequency_threshold, 1, 0)
    # Mask out infrequent attributes
    freq_attributes_dict = {}
    for split in splits:
        cur_attributes = attributes[split]
        freq_attributes = np.where(freq_attributes_one_hot == 1, cur_attributes, 0)

        # Sanity checks
        discarded_attributes_idxs = np.nonzero(np.where(train_counts < frequency_threshold, 1, 0))[0]
        kept_attributes_idxs = np.nonzero(train_counts > frequency_threshold)[0]
        assert (kept_attributes_idxs == np.nonzero(freq_attributes_one_hot)[0]).all()

        zeroed_ctr = 0
        ctr = 0

        for idx, (orig, new) in enumerate(zip(cur_attributes, freq_attributes)):
            # print(orig
            if not (orig == new).all():
                orig_idxs = np.nonzero(orig)[0]
                new_idxs = np.nonzero(new)[0]
                # Assert new idxs ONLY contains the kept attributes and none of discarded
                assert len(np.intersect1d(new_idxs, discarded_attributes_idxs)) == 0
                assert len(np.intersect1d(new_idxs, kept_attributes_idxs)) == len(new_idxs)
                # Assert overlap with original indices is equal to new indices
                assert (np.intersect1d(orig_idxs, new_idxs) == new_idxs).all()
                if len(new_idxs) == 0:
                    zeroed_ctr += 1
                ctr += 1
        print("{} examples have no more attributes".format(zeroed_ctr))
        print("{}/{} examples affected".format(ctr, len(cur_attributes)))
        freq_attributes_dict[split] = freq_attributes

    return freq_attributes_dict, freq_attributes_one_hot