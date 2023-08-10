import os, sys
import stat
import torch
import pickle
import shutil

# To be run following `save_features.py`

logreg_path = 'saved/PlacesCategoryClassification/sklearn_logreg/linear_saga_l2_0.005.pickle'
features_path = 'saved/places_model_ade20k_scene_labeled_features/0810_104502/features.pth'
overwrite = False

def get_model_outputs(model_path,
                      features_path,
                      save_dir=None,
                      overwrite=False):
    '''
    Given logistic regression model in pickle file and features, obtain model outputs

    Arg(s):
        model_path : str
            path to sklearn logistic regression object in .pickle file
        features_path : str
            path to .pth file containing dict[split] = np.array of features
        save_dir : str or None 
            if None, save in directory of features_path
    '''

    if save_dir is None:
        save_dir = os.path.dirname(features_path)

    output_save_path = os.path.join(save_dir, 'outputs_predictions.pth')
    # Check if need to overwrite
    if os.path.exists(output_save_path) and not overwrite:
        print("Outputs exist at {}. Aborting.".format(output_save_path))
        return
    
    # Load model
    try:
        with open(model_path, 'rb') as file:
            logreg_model = pickle.load(file)
    except: 
        raise ValueError("Unable to unpickle file at {}".format(model_path))
    # Copy model to save dir
    copy_model_path = os.path.join(save_dir, os.path.basename(model_path))
    if not os.path.exists(copy_model_path) or overwrite:
        os.chmod(copy_model_path, stat.S_IWRITE)
        shutil.copy(model_path, copy_model_path)
        os.chmod(copy_model_path, stat.S_IWRITE)

    # Load features
    try:
        features = torch.load(features_path)
    except:
        raise ValueError("Unable to load features from {}".format(features_path))
    
    # Data structure to hold outputs for each split
    all_outputs = {}
    splits = features.keys()
    for split in splits:
        split_features = features[split]
        
        try:
            outputs = logreg_model.decision_function(split_features)
            probabilities = logreg_model.predict_proba(split_features)
            predictions = logreg_model.predict(split_features)
        except Exception as e:
            raise ValueError(e)
        split_outputs = {
            'outputs': outputs,
            'probabilities': probabilities,
            'predictions': predictions
        }
        all_outputs[split] = split_outputs

    # Save outputs
    torch.save(all_outputs, output_save_path)
    print("Saved outputs to {}".format(output_save_path))
    

if __name__ == "__main__":
    get_model_outputs(
        model_path=logreg_path,
        features_path=features_path
    )