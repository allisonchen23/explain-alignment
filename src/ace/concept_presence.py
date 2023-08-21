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
                 log_path=None,
                 debug=False):
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

        # Save paths
        self.save_pv_path_template = os.path.join(
            self.save_dir, 
            'presence_vectors', 
            '{}_{}'.format(self.pooling_mode, self.presence_threshold), 
            '{}_{}presence_vectors.pth')
        
        self.debug = debug
        if self.debug:
            self.n_debug = 100

    def _load_split_features(self, split, overwrite=False):
        split_paths = self.features_paths[split]

        split_features_path = os.path.join(os.path.dirname(self.features_dir), '{}_superpixel_features.pth'.format(split))
        if os.path.exists(split_features_path) and not overwrite and not self.debug:
            split_features = torch.load(split_features_path)
            return split_features
        
        if self.debug:
            split_paths = split_paths[:self.n_debug]
        
        split_features = [torch.load(path) for path in tqdm(split_paths)]
        
        if not self.debug:
            torch.save(split_features, split_features_path)

        # if self.debug:
        #     split_features = split_features[:100]
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
    
    def get_all_concepts_presence(self,
                                  features,
                                  concept_cavs_dict):
        concepts_presence = []
        for concept_name in self.concept_names:
            cavs = concept_cavs_dict[concept_name]
            image_concept_presence = self.get_one_concept_presence(
                concept_cavs=cavs,
                features=features
            )
            concepts_presence.append(image_concept_presence)
        concepts_presence = np.stack(concepts_presence, axis=-1)

        return concepts_presence


    def get_split_all_concept_presence(self, split, overwrite=False):
        save_pv_path = self.save_pv_path_template.format(
            split, self.n_debug if self.debug else '')
        ensure_dir(os.path.dirname(save_pv_path))

        print(save_pv_path)
        if os.path.exists(save_pv_path) and not overwrite:
            all_presence_vectors = torch.load(save_pv_path)
            return all_presence_vectors
        
        if split not in self.features:
            informal_log("Loading features from {} split".format(split), self.log_path)
            split_features = self._load_split_features(split=split)
        split_paths = self.features_paths[split]
        # Iterate through all concepts
        all_concept_presences = []
        split_paths_features = list(zip(split_paths, split_features))

        # Load in all cavs
        informal_log("Loading all CAVs for {} concepts".format(len(self.concept_names)), self.log_path)
        cavs = {}
        for concept_name in tqdm(self.concept_names):
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
            cavs[concept_name] = concept_cavs

        all_presence_vectors = []
        informal_log("Iterating through {} split calculating presence vectors...".format(split))
        for features_path, features in tqdm(split_paths_features):
            concepts_presence = self.get_all_concepts_presence(
                features=features,
                concept_cavs_dict=cavs)
            all_presence_vectors.append(concepts_presence)

        if self.pooling_mode is not None:
            all_presence_vectors = np.stack(all_presence_vectors, axis=0)
        
        informal_log("Saving {} presence vectors from {} split to {}".format(
            len(all_presence_vectors), split, save_pv_path
        ), self.log_path)
        torch.save(all_presence_vectors, save_pv_path)
        return all_presence_vectors
