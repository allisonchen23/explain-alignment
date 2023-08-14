import argparse
import os, sys
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')
from utils.attribute_utils import get_one_hot_attributes
from utils.utils import informal_log, ensure_dir, write_lists
from save_attributes import get_frequent_attributes
import ace.cav as cav

N_MAX_CAVS = 615 # Number of attributes with > 1 sample labeled
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
        self.cav_save_dir = os.path.join(cav_save_dir, 'cavs')
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
        '''
        Train n_trials CAVs for a specific attribute

        Returns: 
            list[CAV] : list of CAVs, one for each trial
        '''
        # Create save dir for this attribute
        cav_dir = os.path.join(self.cav_save_dir, attr_id)
        informal_log("Saving/Loading CAVs in {}".format(cav_dir), self.log_path)

        # Get idxs of positive and negative samples
        positive_idxs = np.nonzero(train_labels)[0]
        negative_idxs = np.nonzero(1 - train_labels)[0]
        positive_features = train_features[positive_idxs]
        negative_features = train_features[negative_idxs]

        n_training_samples = min(len(positive_idxs), len(negative_idxs))

        cavs = []
        # Random trials
        for trial_idx in range(n_trials):
            # Balance out number positive and negative activations
            if n_training_samples == len(positive_idxs):
                positive_training_features = positive_features
                negative_training_features = negative_features[
                    np.random.choice(len(negative_features), size=n_training_samples, replace=False)]
            else:
                positive_training_features = positive_features[
                    np.random.choice(len(positive_features), size=n_training_samples, replace=False)]
                negative_training_features = negative_features
            
            random_name = 'random_trial_{}'.format(trial_idx)
            activations = {
                attr_id: {
                    'avgpool': positive_training_features
                },
                random_name: {
                    'avgpool': negative_training_features
                }
            }
            cav_instance = cav.get_or_train_cav(
                concepts=[attr_id, random_name],
                bottleneck='avgpool',
                acts=activations,
                cav_dir=cav_dir,
                cav_hparams=self.cav_hparams,
                overwrite=overwrite,
                save_linear_model=True,
                Cs_hparam_search=Cs,
                log_path=self.log_path
            )
            cavs.append(cav_instance)
        return cavs
        
    def train_cavs(self, 
                   n_attributes=None,
                   n_trials=20):
        # n_total_attributes = len(self.sorted_attr_df)

        # Set n_attributes to a valid value
        if n_attributes is None or n_attributes <= 0 or n_attributes > N_MAX_CAVS:
            n_attributes = N_MAX_CAVS
            informal_log("Received invalid argument for n_attributes ({}). Setting to {}".format(
                n_attributes, N_MAX_CAVS), self.log_path)
            
        # Only look at top attributes
        sorted_attr_df = self.sorted_attr_df.head(n_attributes)
        # Get array of attribute idxs and names
        attribute_idxs = sorted_attr_df['attribute_idxs'].to_numpy()
        attribute_names = sorted_attr_df['names'].to_list()
        
        # Get attribute dense labels
        attribute_labels = get_frequent_attributes(
            save_dir=self.attribute_save_dir,
            n_attributes = n_attributes
        )
        # get training split
        train_attribute_labels = attribute_labels['train']
        
        # Store list of paths
        cav_dir_paths = []

        # Iterate through all attributes
        for idx, attr_idx in enumerate(attribute_idxs):
            
            attr_name = attribute_names[idx].replace(' ', '_')
            attr_id = '{}_{}'.format(attr_idx, attr_name)
            attr_train_labels = train_attribute_labels[:, idx]

            informal_log("Training CAVs for {}".format(attr_name), self.log_path)
            self._train_cavs_for_attribute(
                attr_id=attr_id,
                train_features=self.features['train'],
                train_labels=attr_train_labels,
                n_trials=n_trials
            )
            cav_dir_path = os.path.join(self.cav_save_dir, attr_id)
            cav_dir_paths.append(cav_dir_path)
        
        # Write out paths to cav directories
        cav_dir_paths_path = os.path.join(os.path.dirname(self.cav_save_dir), 'cav_dirs_{}.txt'.format(n_attributes))
        write_lists(cav_dir_paths, cav_dir_paths_path)
        informal_log("Saved paths to cav directories to {}".format(cav_dir_paths_path), self.log_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_attributes', type=int, default=1, help='Number of attributes to calculate CAVs for. Starting with most to least frequent.')
    parser.add_argument('--cav_save_dir', type=str, default='saved/places_model_ade20k_scene_labeled_features/labeled_cavs',
                        help="Directory to save cavs inside. Actual cavs will be saved in a subdirectory 'cavs'")
    args = parser.parse_args()

    features_path = 'saved/places_model_ade20k_scene_labeled_features/0810_104502/features.pth'
    sorted_attributes_csv_path = 'data/ade20k/scene_annotated/sorted_attributes.csv'
    attribute_save_dir = 'data/ade20k/scene_annotated'
    log_path = os.path.join(args.cav_save_dir, 'log.txt')
    
    labeled_cavs = LabeledCavs(
        features_path=features_path,
        sorted_attr_csv_path=sorted_attributes_csv_path,
        cav_save_dir=args.cav_save_dir,
        attribute_save_dir=attribute_save_dir,
        log_path=log_path
    )
    
    labeled_cavs.train_cavs(
        n_attributes=args.n_attributes
    )