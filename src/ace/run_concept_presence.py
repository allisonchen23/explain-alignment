import os, sys
import argparse
import torch

from concept_presence import ConceptPresence
sys.path.insert(0, 'src')
from utils.utils import informal_log

def main(n_samples, 
         pooling_mode,
         presence_threshold,
         split=None,
         debug=False):
    checkpoint_dir = 'saved/ace/n_{}'.format(n_samples)
    concept_key = 'concepts-K_27-min_20-max_40'
    concept_dictionary_path = os.path.join(checkpoint_dir, 'saved/{}/concept_dictionary.pth'.format(concept_key))
    splits = ['train', 'val', 'test']
    features_paths = [
        '/n/fs/ac-alignment/explain-alignment/saved/ADE20K/0501_105640/{}_features.pth'.format(split)
        for split in splits
    ]

    updated_features_dir = 'data/ade20k/ace/full_ade20k/superpixel_features'
    image_labels_path = 'data/ade20k/full_ade20k_imagelabels.pth'
    save_pv = True
    overwrite_pv = False
    # presence_threshold = 0.5
    # pooling_mode = 'average'

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', required=True, type=int,
                        help='Number of samples used to calculate ACE concepts')
    parser.add_argument('--pooling_mode', type=str, required=True,
                        help='Type of pooling to determine concept presence per image')
    parser.add_argument('--presence_threshold', type=float, required=True,
                        help='Threshold for concept to be deemed present')
    parser.add_argument('--split', type=str, default=None,
                        help='Which split to calculate presence vectors for')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        pooling_mode=args.pooling_mode,
        presence_threshold=args.presence_threshold,
        split=args.split,
        debug=args.debug
    )