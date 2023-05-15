import numpy as np
import os, sys
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, 'src')
from utils.utils import informal_log

def get_one_hot_attributes(data, paths, n_attr, splits=['train', 'val', 'test']):
    attributes = {}
    # for split in splits:
    #     attributes[split] = []

    for split in splits:
        attributes[split] = []
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

def hyperparam_search_l1(train_features, train_labels, val_features, val_labels,
                      Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5]):
    best_clf = None
    best_acc = 0

    for c in Cs:
        clf = LogisticRegression(solver='liblinear', C=c, penalty='l1')
        clf.fit(train_features, train_labels)
        score = clf.score(val_features, val_labels)
        if score>best_acc:
            best_acc = score
            best_clf = clf
            print("Best accuracy: {} Regularization: {}".format(score, c))

    return best_clf

def hyperparam_search(train_features,
                                  train_labels,
                                  val_features,
                                  val_labels,
                                  regularization,
                                  solver,
                                  scaler=None,
                                  Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5],
                                  log_path=None):
    best_clf = None
    best_acc = 0

    if scaler is not None:
        scaler.fit(train_features)
        print("Scaler parameters: {}".format(scaler.get_params()))
        train_features = scaler.transform(train_features)
        val_features = scaler.transform(val_features)
    for c in Cs:
        clf = LogisticRegression(solver=solver, C=c, penalty=regularization)
        clf.fit(train_features, train_labels)
        score = clf.score(val_features, val_labels)
        if score>best_acc:
            best_acc = score
            best_clf = clf
            informal_log("Best accuracy: {} Regularization: {}".format(score, c), log_path)

    return best_clf

def partition_paths_by_congruency(explainer_predictions,
                                  model_predictions,
                                  paths):
    '''
    Given list or arrays of explainer and model predictions, partition paths based on if predictions align

    Arg(s):
        explainer_predictions : N-length np.array
            predictions output by the explainer model
        model_predictions : N-length np.array
            predictions output by the model
        paths : N-length list
            paths of images corresponding to each data point

    Returns:
        dictionary : dict[str] : list
            key: 'congruent' or 'incongruent'
            value: list of paths
    '''
    n_samples = len(paths)
    assert len(explainer_predictions) == n_samples
    assert len(model_predictions) == n_samples, "Length of model predictions {} doesn't match n_samples {}".format(
        len(model_predictions), n_samples
    )

    incongruent_paths = []
    congruent_paths = []

    for explainer_prediction, model_prediction, path in tqdm(zip(
        explainer_predictions, model_predictions, paths
    )):
        if explainer_prediction == model_prediction:
            congruent_paths.append(path)
        else:
            incongruent_paths.append(path)

    return {
        'congruent': congruent_paths,
        'incongruent': incongruent_paths
    }