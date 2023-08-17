import os, sys
import numpy as np
import pickle
import torch
from tqdm import tqdm

sys.path.insert(0, 'src')
from utils.utils import ensure_dir, save_torch, informal_log, read_lists, write_lists, save_image


class ConceptPresence():
    def __init__(self,
                 concept_dictionary,
                 checkpoint_dir,
                 concept_key,
                 features,
                 image_labels_path,
                 features_dir,
                 splits=['train', 'val', 'test'],
                 presence_threshold=0.5,
                 pooling_mode=None,
                 log_path=None):
        '''
        Given a concept dictionary, load CAVs, and create concept-presence vectors for all features

        Arg(s):
            concept_dictionary : dict
                {
                    'concept1_name': {
                        'bottleneck1': {
                            'features': np.array,
                            'image_numbers': np.array,
                            'accuracies': np.array,
                        },
                        'bottleneck2': {
                            'features': np.array,
                            'image_numbers': np.array,
                            'accuracies': np.array,
                        }
                    }
                }
            features : dict[str : list[np.array]]
                str is splits
                features of different splits
                np.array : N x D dimensions
            features_paths : list[str]
                List of paths to each image's features stored in .pth files as np.arrays
            checkpoint_dir : str
                same checkpoint dir as from run_ace.py
                CAVs should be stored in 'saved/<concept_key>/cavs'
            present_threshold : float
                Proportion of CAVs that must count this concept to be present to truly be present
        '''

        self.concept_dictionary = concept_dictionary
        self.concept_names = list(self.concept_dictionary.keys())

        # Directory stuff, to be consistent with my implementation of ACE
        self.checkpoint_dir = checkpoint_dir
        self.concept_key = concept_key
        self.save_dir = os.path.join(self.checkpoint_dir, 'saved', self.concept_key)
        self.cav_dir = os.path.join(self.save_dir, 'cavs')
        self.log_path = log_path

        self.features = features
        self.features_dir = features_dir
        self.presence_threshold = presence_threshold
        self.pooling_mode = pooling_mode
        self.splits = splits
        assert len(self.features) == len(self.splits)

        # Sanity check that all elements in cav_dir are in concept names
        cav_dir_contents = os.listdir(self.cav_dir)
        assert set(cav_dir_contents) == set(self.concept_names)

        # Load image_labels
        image_labels = torch.load(image_labels_path)

        # Create dictionary of split : paths to features
        self.features_paths = {}
        for split in splits:
            split_image_paths = image_labels[split]
            split_features_paths = []
            for image_path in split_image_paths:
                image_id = os.path.basename(image_path).split('.')[0]
                features_path = os.path.join(self.features_dir, '{}_features.pth'.format(image_id))
                split_features_paths.append(features_path)
            self.features_paths[split] = split_features_paths
        
        # self.features = self._load_features()
        self.features = {}

    def _load_split_features(self, split):
        split_paths = self.features_paths[split]
        split_features = [torch.load(path) for path in tqdm(split_paths)]
        return split_features
    
    def get_one_concept_presence(self,
                                 concept_cavs,
                                 features):
        '''
        Given a list of CAVs for 1 concept, and np.array of features from patches, output concept presence/absence

        Arg(s):
            concept_cavs : list[CAV]
                C-length repeated trials trained for a single CAV
            features : N x D np.array
                Features from the image.
                If using with ACE, these should be the features from the superpixel patches for a single image
            pooling : None or str
                If None, return a N-dim np.array of {0, 1} for concept presence in each features
                If 'max': return maximum concept presence across features
                If 'average': return average concept presence -> threshold with 0.5
        
        Returns: 
            N-dim np.array of {0, 1} OR
            int {0, 1}
        '''
        concept_presences = []
        for cav in concept_cavs:
            lm = cav['linear_model']

            # N-dim vector
            concept_predictions = lm.predict(features)
            # This is because how CAV is implemented
            # First concept corresponds with the target concept, second is a random concept
            concept_present = 1 - concept_predictions
            concept_presences.append(concept_present)

        # Take average across trials for each feature
        concept_presences = np.stack(concept_presences, axis=1) # N x C vector
        concept_presence = np.mean(concept_presences, axis=1) # N-dim vector
        
        if self.pooling_mode is None:
            return np.where(concept_presence > self.presence_threshold, 1, 0)
        elif self.pooling_mode == 'max':
            concept_presence = np.max(concept_presence)
        elif self.pooling_mode == 'average':
            concept_presence = np.mean(concept_presence)
        if concept_presence > self.presence_threshold:
            return 1
        else:
            return 0

        
    def get_split_one_concept_presence(self,
                                        split_features,
                                        concept_cavs,
                     save=True,
                     overwrite=False):
        '''
        
        '''
        

        split_concept_presence = []
        for image_features in split_features:
            image_concept_presence = self.get_one_concept_presence(
                concept_cavs=concept_cavs,
                features=image_features,
                # pooling=self.pooling_mode,
            ) # either a N_features-dim vector (one for each patch) or a binary value
            split_concept_presence.append(image_concept_presence)
        return split_concept_presence

    def get_split_all_concept_presence(self, split):
        if split not in self.features:
            informal_log("Loading features from {} split".format(split), self.log_path)
            self.features[split] = self._load_split_features(split=split)
        
        # Iterate through all concepts
        all_concept_presences = []
        for concept_name in self.concept_names:
            concept_cav_dir = os.path.join(self.cav_dir, concept_name)
            cav_paths = [
                os.path.join(concept_cav_dir, cav_name) 
                for cav_name in os.listdir(concept_cav_dir)
            ]
            # Load CAVs for this concept
            concept_cavs = []
            for cav_path in cav_paths:
                with open(cav_path, 'rb') as file:
                    cav_instance = pickle.load(file)
                concept_cavs.append(cav_instance)
            split_concept_presence = self.get_split_one_concept_presence(
                concept_cavs=concept_cavs,
                split_features=self.features[split]
            )
            all_concept_presences.append(split_concept_presence)

        if self.pooling_mode is None:
            pass
        else:
            all_concept_presences = np.stack(all_concept_presences, axis=1)
        return

    def old(self):
        '''
        Determine if concepts are present or absent from each feature vector

        Returns: 
            list[np.array]
                Same number of elements as self.features
                Each element is an (self.features.shape[0] x len(self.concept_names)) np.array
                    corresponding to concept presence.
        '''

        presence_vectors_path = os.path.join(self.save_dir, 'presence_vectors.pth')
        if not overwrite and os.path.exists(presence_vectors_path):
            pv = torch.load(presence_vectors_path)
            informal_log("Loaded presence vectors from {}".format(presence_vectors_path),
                         self.log_path, timestamp=True)
            return pv
        # pv stands for presence vectors
        pv = {}
        informal_log("Creating presence vectors...", self.log_path, timestamp=True)
        for split, split_features in zip(self.splits, self.features):
            informal_log("Processing {} features...".format(split), self.log_path, timestamp=True)

            split_pv = []
            for concept_name in self.concept_names:
                concept_cav_dir = os.path.join(self.cav_dir, concept_name)
                cav_paths = [
                    os.path.join(concept_cav_dir, cav_name) 
                    for cav_name in os.listdir(concept_cav_dir)
                ]
                trial_pvs = []

                # Get concept predictions for all trained CAVs (against different negative examples)
                for cav_path in cav_paths:
                    with open(cav_path, 'rb') as file:
                        cav_instance = pickle.load(file)
                    lm = cav_instance['linear_model']

                    concept_predictions = lm.predict(split_features)
                    # This is because how CAV is implemented
                    # First concept corresponds with the target concept, second is a random concept
                    concept_present = 1 - concept_predictions
                    trial_pvs.append(concept_present)
                
                trial_pvs = np.stack(trial_pvs, axis=1) # N_trials X N_samples in features
                trial_pv = np.mean(trial_pvs, axis=1) # N_samples vector with proportion of CAVs that said concept is present
                trial_pv = np.where(trial_pv > self.presence_threshold, 1, 0)

                split_pv.append(trial_pv)
            
            # List of length n_concepts, make into array of N_samples x N_concepts
            split_pv = np.stack(split_pv, axis=1)
            pv[split] = split_pv
        
        assert len(pv) == len(self.features)

        if save:
            torch.save(pv, presence_vectors_path)
            informal_log("Saved presence vectors to {}".format(presence_vectors_path), self.log_path, timestamp=True)
        return pv


