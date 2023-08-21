import os, sys
import argparse
import torch

from concept_presence import ConceptPresence
sys.path.insert(0, 'src')
from utils.utils import informal_log

def main(n_samples, 
         n_concepts,
         pooling_mode,
         presence_threshold,
         split=None,
         debug=False):
    checkpoint_dir = 'saved/ace/n_{}'.format(n_samples)
    concept_key = 'concepts-K_{}-min_20-max_40'.format(n_concepts)
    concept_dictionary_path = os.path.join(checkpoint_dir, 'saved/{}/concept_dictionary.pth'.format(concept_key))
    
    if not os.path.exists(concept_dictionary_path):
        raise ValueError("Unable to find concept dictionary at {}".format(concept_dictionary_path))
    
    splits = ['train', 'val', 'test']
    features_paths = [
        '/n/fs/ac-alignment/explain-alignment/saved/ADE20K/0501_105640/{}_features.pth'.format(split)
        for split in splits
    ]

    updated_features_dir = 'data/ade20k/ace/full_ade20k/superpixel_features'
    image_labels_path = 'data/ade20k/full_ade20k_imagelabels.pth'
    save_pv = True
    overwrite_pv = False

    # Load in features from each split
    features = []
    for path in features_paths:
        features.append(torch.load(path)['features'])

    concept_dictionary = torch.load(concept_dictionary_path)
    log_path = os.path.join(os.path.dirname(concept_dictionary_path), 'concept_presence_log.txt')
    informal_log("---***---\nObtaining ConceptPresenceVectors", log_path,timestamp=True)

    cp = ConceptPresence(
        concept_dictionary=concept_dictionary,
        checkpoint_dir=checkpoint_dir,
        concept_key=concept_key,
        features=features,
        features_dir=updated_features_dir,
        image_labels_path=image_labels_path,
        splits=['train', 'val', 'test'],
        presence_threshold=presence_threshold,
        pooling_mode=pooling_mode,
        log_path=log_path,
        debug=debug
    )
    if split is None:
        for split in splits:
            cp.get_split_all_concept_presence(split=split)
    else:
        cp.get_split_all_concept_presence(split=split)
    # cp.get_presence(
    #     save=save_pv,
    #     overwrite=overwrite_pv
    # )

def consolidate(n_samples, 
                n_concepts,
                pooling_mode,
                presence_threshold):
    checkpoint_dir = 'saved/ace/n_{}'.format(n_samples)
    concept_key = 'concepts-K_{}-min_20-max_40'.format(n_concepts)
    concept_dictionary_path = os.path.join(checkpoint_dir, 'saved/{}/concept_dictionary.pth'.format(concept_key))
    if not os.path.exists(concept_dictionary_path):
        raise ValueError("Unable to find concept dictionary at {}".format(concept_dictionary_path))
    
    splits = ['train', 'val', 'test']
    features_paths = [
        '/n/fs/ac-alignment/explain-alignment/saved/ADE20K/0501_105640/{}_features.pth'.format(split)
        for split in splits
    ]

    updated_features_dir = 'data/ade20k/ace/full_ade20k/superpixel_features'
    image_labels_path = 'data/ade20k/full_ade20k_imagelabels.pth'

    # presence_threshold = 0.5
    # pooling_mode = 'average'

    # Load in features from each split
    features = []
    for path in features_paths:
        features.append(torch.load(path)['features'])

    concept_dictionary = torch.load(concept_dictionary_path)
    # log_path = os.path.join(os.path.dirname(concept_dictionary_path), 'concept_presence_log.txt')
    cp = ConceptPresence(
        concept_dictionary=concept_dictionary,
        checkpoint_dir=checkpoint_dir,
        concept_key=concept_key,
        features=features,
        features_dir=updated_features_dir,
        image_labels_path=image_labels_path,
        splits=splits,
        presence_threshold=presence_threshold,
        pooling_mode=pooling_mode,
        log_path=None,
    )

    pv_path_template = cp.save_pv_path_template

    pvs = {}
    for split in splits:
        split_path = pv_path_template.format(split, "")
        if not os.path.exists(split_path):
            raise ValueError('{} does not exist'.format(split_path))
        pvs[split] = torch.load(split_path)

    pv_dir = os.path.dirname(pv_path_template)
    pooling_params = os.path.basename(pv_dir)
    consolidated_save_path = os.path.join(pv_dir, '{}_dense_one_hot_attributes.pth'.format(pooling_params))
    torch.save(pvs, consolidated_save_path)
    return
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', required=True, type=int,
                        help='Number of samples used to calculate ACE concepts')
    parser.add_argument('--n_concepts', required=True, type=int,
                        help='Number of concepts calculated')
    parser.add_argument('--pooling_mode', type=str, required=True,
                        help='Type of pooling to determine concept presence per image')
    parser.add_argument('--presence_threshold', type=float, required=True,
                        help='Threshold for concept to be deemed present')
    parser.add_argument('--split', type=str, default=None,
                        help='Which split to calculate presence vectors for')
    parser.add_argument('--mode', type=str, default='main',
                        help='What function to run')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    modes_available = ['main', 'consolidate']

    if args.mode not in modes_available:
        raise ValueError("Unrecognized mode '{}'. Try one of {}".format(args.mode, modes_available))
    
    if args.mode == 'main':
        main(
            n_samples=args.n_samples,
            n_concepts=args.n_concepts,
            pooling_mode=args.pooling_mode,
            presence_threshold=args.presence_threshold,
            split=args.split,
            debug=args.debug
        )
    elif args.mode == 'consolidate':
        consolidate(
            n_samples=args.n_samples,
            pooling_mode=args.pooling_mode,
            presence_threshold=args.presence_threshold
        )