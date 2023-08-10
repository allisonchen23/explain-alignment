import os, sys
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')
from utils.attribute_utils import get_one_hot_attributes
from utils.utils import informal_log, ensure_dir
from save_attributes import get_frequent_attributes
import ace.cav as cav

class LabeledCavs(object):
    def __init__(self,
                 features_path,
                 sorted_attr_csv_path,
                 cav_save_dir,
                 attribute_save_dir=None,
                 log_path=None):
        
        self.features = torch.load(features_path)
        self.sorted_attr_df = pd.read_csv(sorted_attr_csv_path)

        if attribute_save_dir is None:
            self.attribute_save_dir = os.path.dirname(sorted_attr_csv_path)
        else:
            self.attribute_save_dir = attribute_save_dir

        # CAV related
        self.cav_save_dir = cav_save_dir
        ensure_dir(cav_save_dir)
        self.cav_hparams = {
            'model_type': 'logistic',
            'alpha': None,
            'model_params': {}
        }

        self.log_path = log_path

    def _train_cavs_for_attribute(self,
                                  attr_id,
                                  train_features,
                                  train_labels,
                                  n_trials,
                                  overwrite=False,
                                  Cs=[0.001, 0.01, 0.1, 1, 5]):
        # Create save dir for this attribute
        cav_dir = os.path.join(self.cav_save_dir, attr_id)
        
        # Get idxs of positive and negative samples
        positive_idxs = np.nonzero(train_labels)[0]
        negative_idxs = np.nonzero(1 - train_labels)[0]
        positive_features = train_features[positive_idxs]
        negative_features = train_features[negative_idxs]

        n_training_samples = min(len(positive_idxs), len(negative_idxs))

        # Balance out number of positive and negative features
        if n_training_samples == len(positive_idxs):
            positive_training_features = positive_features
            negative_training_features = negative_features[
                np.random.choice(len(negative_features), size=n_training_samples, replace=False)]
        else:
            positive_training_features = positive_features[
                np.random.choice(len(positive_features), size=n_training_samples, replace=False)]
            negative_training_features = negative_features

        # Do a hparam search for C
        best_c = -1
        best_accuracy = -1
        accuracies = []
        for C in Cs:
            random_name = 'random_hparam_C_{}'.format(C)
            activations = {
                attr_id: {
                    'avgpool': positive_training_features
                },
                random_name: {
                    'avgpool': negative_training_features
                }
            }
            cav_hparams = self.cav_hparams.copy()
            cav_hparams['model_params'].update({'C': C})
            cav_instance = cav.get_or_train_cav(
                concepts=[attr_id, random_name],
                bottleneck='avgpool',
                acts=activations,
                cav_dir=cav_dir,
                cav_hparams=cav_hparams,
                overwrite=True,
                save_linear_model=True,
                log_path=self.log_path
            )
            cav_accuracy = cav_instance.accuracies['overall']
            accuracies.append(cav_accuracy)
            if cav_accuracy > best_accuracy:
                best_accuracy = cav_accuracy
                best_c = C
            

        # Once we have found the best C, run multiple trials
        # cavs = []
        # for trial_idx in n_trials:
            # get random negative samples
            # do something like ace._calculate_cav
            # train cav
            # cavs.append(cav)
        # save cavs
        pass
    def train_cavs(self, 
                   n_attributes=None,
                   n_trials=50):
        n_total_attributes = len(self.sorted_attr_df)

        # Set n_attributes to a valid value
        if n_attributes is None or n_attributes <= 0 or n_attributes > n_total_attributes:
            n_attributes = n_total_attributes
            informal_log("Received invalid argument for n_attributes ({}). Setting to {}".format(
                n_attributes, n_total_attributes), self.log_path)
            
        # Only look at top attributes
        sorted_attr_df = self.sorted_attr_df.head(n_attributes)
        # Get array of attribute idxs and names
        attribute_idxs = sorted_attr_df['attribute_idxs'].to_numpy()
        attribute_names = sorted_attr_df['names'].to_list()
        # Get list of attribute idxs that we do not have CAVs for
        untrained_attr_idxs = []
        for attr_idx, attr_name in zip(attribute_idxs, attribute_names):
            cav_dir = os.path.join(self.cav_save_dir, '{}_{}'.format(attr_idx, attr_name.replace(' ', '_')))
            if not os.path.exists(cav_dir):
                untrained_attr_idxs.append(attr_idx)
        if len(untrained_attr_idxs) == 0:
            return
        
        # Get attribute dense labels
        attribute_labels = get_frequent_attributes(
            save_dir=self.attribute_save_dir,
            n_attributes = n_attributes
        )
        print(attribute_labels.keys())
        # get training split
        train_attribute_labels = attribute_labels['train']
        
        # Iterate through all attributes
        for idx, attr_idx in enumerate(attribute_idxs):
            
            attr_name = attribute_names[idx]
            attr_id = '{}_{}'.format(attr_idx, attr_name)
            attr_train_labels = train_attribute_labels[:, idx]

            informal_log("Training CAVs for {}".format(attr_name))
            self._train_cavs_for_attribute(
                attr_id=attr_id,
                train_features=self.features['train'],
                train_labels=attr_train_labels,
                n_trials=n_trials
            )
        # for each attribute:
            # train_cav
        pass
        

if __name__ == "__main__":
    features_path = 'saved/places_model_ade20k_scene_labeled_features/0810_104502/features.pth'
    sorted_attributes_csv_path = 'data/ade20k/scene_annotated/sorted_attributes.csv'
    attribute_save_dir = 'data/ade20k/scene_annotated'
    cav_save_dir = 'saved/places_model_ade20k_scene_labeled_features/cavs'
    labeled_cavs = LabeledCavs(
        features_path=features_path,
        sorted_attr_csv_path=sorted_attributes_csv_path,
        cav_save_dir=cav_save_dir,
        attribute_save_dir=attribute_save_dir
    )
    labeled_cavs.train_cavs()