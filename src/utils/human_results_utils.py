import json
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

sys.path.insert(0, 'src')
from utils.utils import ensure_dir, list_to_dict, read_lists
# functionality used to be in clean_process_df.ipynb

# Define constants
MEASUREMENT_COLUMN_NAMES = ['selectedAttrs', 'attrUncs']
TASK_METADATA_COLUMN_NAMES = ['filename', 'task', 'concept_group']
SCENE_CATEGORIES_PATH = os.path.join('data', 'ade20k', 'scene_categories.txt')

def clean_df(df):
    measurement_df = df[MEASUREMENT_COLUMN_NAMES]
    metadata_df = df.drop(MEASUREMENT_COLUMN_NAMES, axis=1)

    # Drop empty rows
    measurement_df = measurement_df.dropna()
    # Drop rows without data in task metadata columns
    metadata_df = metadata_df.dropna(subset=TASK_METADATA_COLUMN_NAMES)

    # Remove columns that are empty
    metadata_df = metadata_df.dropna(axis=1)

    # Assert that the two DFs have the same number of rows
    assert len(metadata_df) == len(measurement_df), "Uneven length data frames. Metadata length: {} Measurement length: {}".format(
        len(metadata_df), len(measurement_df))

    # Reset indices to allow for joining appropriately
    metadata_df = metadata_df.reset_index(drop=True)
    measurement_df = measurement_df.reset_index(drop=True)


    # Join the data frames
    df = pd.concat([metadata_df, measurement_df], axis=1)
    assert len(df) == len(metadata_df)
    return df
    
def clean_dfs(csv_paths):
    dfs = []
    
    for csv_path in csv_paths:
        print("Processing {}".format(os.path.basename(csv_path)))
        df = pd.read_csv(csv_path)
        df = clean_df(df)
        dfs.append(df)
    return dfs
        
def clean_and_merge_csvs(results_dir):
    '''
    Given a directory to where CSVs are stored, return one dataframe of human survey results across all CSV files
    
    Arg(s):
        results_dir : str
            directory of CSV files
    
    Returns:
        pd.DataFrame : merged dataframe from each CSV
    '''
    csv_paths = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            csv_paths.append(os.path.join(results_dir, filename))
    
    # os.listdir() doesn't guarantee any order
    csv_paths = sorted(csv_paths)
    
    dfs = clean_dfs(csv_paths=csv_paths)
    
    # Merge list of dfs
    df = pd.concat(dfs)
    n_samples = len(df)
    print("Total of {} samples".format(n_samples))
    return df
    
    
def calculate_soft_labels_predictions(df,
                                      scene_categories_dict,
                                      save_dir=None):
    '''
    Given a dataframe from all human responses, calculate the human outputs, probabilities, and predictions

    Arg(s):
        df : pd.DataFrame
            dataframe from processed CSVs
        scene_categories_dict : dict[str : int]
            dictionary from string -> index
        save_path : str or None
            (optional) path to save the outputs to 

    Returns:
        dict[str : ]
    '''
    n_categories = len(scene_categories_dict)
    human_probabilities = []
    human_outputs = []
    human_predictions = []
    
    # Iterate through all the uncertainty measurements
    for row in tqdm(df['attrUncs']):
        soft_label = np.zeros(n_categories)
        # Each 'score' item is a dictionary of class and certainty amount
        row = json.loads(row)
        for item in row:
            category = item['label']
            certainty = item['y'] / 100.0
            category_idx = scene_categories_dict[category]
            soft_label[category_idx] = certainty
        label_sum = np.sum(soft_label)
        
        # Normalize to sum to one
        probability = soft_label / label_sum
        # Assert the soft label sums to 1
        assert np.abs(np.sum(probability) - 1.0) < 1e-5
        
        # Add to lists
        human_outputs.append(soft_label)  # unnormalized
        human_probabilities.append(probability)  # normalized to sum to 1
        human_predictions.append(np.argmax(soft_label))  # predicted class

    # Append
    df['human_probabilities'] = human_probabilities
    df['human_outputs'] = human_outputs
    df['human_predictions'] = human_predictions
    print("Calculated human outputs, probabilities, and predictions")
    human_outputs_predictions = {
        'outputs': human_outputs,
        'probabilities': human_probabilities,
        'predictions': human_predictions,
        'filenames': df['filename']
    }
    
    
    if save_dir is not None:
        ensure_dir(save_dir)
        df_save_path = os.path.join(save_dir, 'human_results.csv')
        outputs_save_path = os.path.join(save_dir, 'human_outputs_predictions.pth')

        df.to_csv(df_save_path)
        print("Saved human survey resuls to {}".format(df_save_path))
        torch.save(human_outputs_predictions, outputs_save_path)
        print("Saved human outputs/probabilities/predictions to {}".format(outputs_save_path))
    return human_outputs_predictions
        
    
def get_human_outputs_predictions(results_dir,
                                  save_dir=None,
                                  overwrite=False):
    # Check if files exist, if they do, return the outputs
    df_save_path = os.path.join(save_dir, 'human_results.csv')
    outputs_save_path = os.path.join(save_dir, 'human_outputs_predictions.pth')
    if os.path.exists(df_save_path) and os.path.exists(outputs_save_path) and not overwrite:
        return torch.load(outputs_save_path)
    
    df = clean_and_merge_csvs(results_dir=results_dir)
    scene_categories = read_lists(SCENE_CATEGORIES_PATH)
    scene_categories_dict = list_to_dict(scene_categories)
    human_outputs_predictions = calculate_soft_labels_predictions(
        df=df,
        scene_categories_dict=scene_categories_dict,
        save_dir=save_dir)
    return human_outputs_predictions


# if __name__ == "__main__":
#     results_dir = 'saved/ADE20K/survey_results/test_split/test_soft_labels'
#     save_dir = 'saved/ADE20K/survey_results/test_split'
#     overwrite=False
#     get_human_outputs_predictions(
#         results_dir=results_dir,
#         save_dir=save_dir,
#         overwrite=overwrite
#     )