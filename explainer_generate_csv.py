import argparse
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import torch
from scipy import stats

sys.path.insert(0, 'src')
from utils.utils import read_lists, load_image, ensure_dir, write_lists
from utils.visualizations import histogram, plot, show_image_rows, bar_graph, pie_chart
from utils.df_utils import convert_string_columns
from utils.metric_utils import top_2_confusion, add_confidence, sort_and_bin_df, calculate_bin_agreement, run_feature_importance_trial, correlated_variables, print_summary, filter_df, string_to_list, plot_metric_v_inputs

DATASETS_AVAILABLE = ['cifar', 'ade20k']
HUMAN_SURVEY_RESULTS_DIR = os.path.join('saved', 'ADE20K', 'survey_results', 'ADE20K_soft_labels')
def process_human_survey_csvs(csv_dir):
    MEASUREMENT_COLUMN_NAMES = ['selectedAttrs', 'attrUncs']
    TASK_METADATA_COLUMN_NAMES = ['filename', 'task', 'concept_group']

    csv_paths = []
    # Get paths to human annotation CSVs
    for filename in os.listdir(csv_dir):
        if filename.endswith('csv'):
            csv_paths.append(os.path.join(csv_dir, filename))

    csv_paths = sorted(csv_paths)

    df_list = []
    # For each CSV, get the human soft labels and add to the dataframe list
    for csv_path in csv_paths:
        print("Processing {}".format(os.path.basename(csv_path)))
        df = pd.read_csv(csv_path)
        # Separate dataframe into rows with measurements and with metadata
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

        # Add dataframe to list of dataframes
        df_list.append(df)

    # Concatenate rows of all dataframes together
    df = pd.concat(df_list)
    return df

def save_ade20k_human_out(df, human_output_save_path):
    SCENE_CATEGORIES_PATH = os.path.join('data', 'ade20k', 'scene_categories.txt')

    # Create 2-way dictionary for scene categories
    scene_categories = read_lists(SCENE_CATEGORIES_PATH)
    scene_categories_dict = {}
    for idx, category in enumerate(scene_categories):
        scene_categories_dict[idx] = category
        scene_categories_dict[category] = idx
    n_categories = len(scene_categories)

    # Extract human probabilities
    human_probabilities = []
    human_outputs = []
    human_predictions = []
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
        human_outputs.append(soft_label)

        # Normalize to sum to one
        soft_label = soft_label / label_sum
        # Assert the soft label sums to 1
        assert np.abs(np.sum(soft_label) - 1.0) < 1e-5

        human_probabilities.append(soft_label)
        human_predictions.append(np.argmax(soft_label))

    df['human_probabilities'] = human_probabilities
    df['human_outputs'] = human_outputs
    df['human_predictions'] = human_predictions
    human_outputs_predictions = {
        'outputs': human_outputs,
        'probabilities': human_probabilities,
        'predictions': human_predictions
    }

    # Save human outputs
    ensure_dir(os.path.dirname(human_output_save_path))
    if os.path.exists(human_output_save_path):
        print("File exists at {}".format(human_output_save_path))
    else:
        torch.save(human_outputs_predictions, human_output_save_path)
        print("Saved human outputs to {}".format(human_output_save_path))
    return human_outputs_predictions


def get_outputs(dataset_name: str,
                human_output_path: str,
                model_output_path: str,
                explainer_output_path: str):
    '''
    Given paths to outputs for human, model and explainers, load and reformat to return uniform standard
    Arg(s):

    Returns:
        dict{str : dict{str : np.array}}
            Indexed by agent type -> dictionary indexed by output type -> outputs for test set
    '''
    if dataset_name == 'cifar':
        human_outputs = np.load(human_output_path)
        human_predictions = np.argmax(human_outputs, axis=1)
        human_out = {
            'outputs': human_outputs,
            'probabilities': human_outputs,
            'predictions': human_predictions
        }
        model_out = torch.load(model_output_path)['test']
        explainer_out = torch.load(explainer_output_path)['test']
        outputs = {
            'human': human_out,
            'explainer': explainer_out,
            'model': model_out
        }
        # ground_truth_labels = torch.load(processed_data_path)['test']['predictions']
        return outputs
    elif dataset_name == 'ade20k':
        df = process_human_survey_csvs(csv_dir=HUMAN_SURVEY_RESULTS_DIR)
        # Process or load human outputs
        if not os.path.exists(human_output_path):
            human_out = save_ade20k_human_out(
                human_outputs_save_path=human_output_path)
        else:
            human_out = torch.load(human_output_path)
        outputs = {
            'human': human_out
        }
        # Load all explainer and model outputs on validation set
        explainer_out = torch.load(explainer_output_path)['test']
        model_out = torch.load(model_output_path)
        # Create dictionary from image name -> index of val set
        image_labels_path = 'data/ade20k/full_ade20k_imagelabels.pth'
        image_labels = torch.load(image_labels_path)
        val_images = image_labels['val']
        val_images = [path.split('images/')[-1] for path in val_images]
        val_name_idx_dict = {}
        for idx, image_name in enumerate(val_images):
            val_name_idx_dict[image_name] = idx

        # Pick out explainer and model outputs for selected images

        for name, agent_outputs in zip(['explainer', 'model'], [explainer_out, model_out]):
            cur_out = {}
            for output_type in ['outputs', 'probabilities', 'predictions']:
                cur_outputs = agent_outputs[output_type]
                accumulator = []
                for image_name in df['filename']:
                    val_idx = val_name_idx_dict[image_name]
                    cur_item = cur_outputs[val_idx]
                    accumulator.append(cur_item)
                cur_out[output_type] = accumulator
                print("acc len: {}".format(len(accumulator)))
            outputs[name] = cur_out
        return outputs


    else:
        raise ValueError("Dataset '{}' not yet supported".format(dataset_name))

def add_outputs(df, outputs):
    # Add human, model, explainer outputs/probabilities/predictions
    for agent, output in outputs.items():
        for out_type in ['outputs', 'probabilities', 'predictions']:
            df['{}_{}'.format(agent, out_type)] = list(output[out_type])
    return df

def add_uncertainty(df):
    # Define functions to calculate each metric
    metric_fns = [
        ('t2c', top_2_confusion),
        ('entropy', stats.entropy),
        ('top_confidence', add_confidence)]
    agents = ['human', 'explainer', 'model']

    for agent in agents:
        for metric_name, metric_fn in metric_fns:
            if metric_name == 't2c':
                inputs = np.stack(df['{}_outputs'.format(agent)].to_numpy(), axis=0)
                t2c = metric_fn(inputs)
                df['{}_{}'.format(agent, metric_name)] = t2c
                min_t2c = np.amin(t2c)
                max_t2c = np.amax(t2c)
                scaled_t2c = (t2c - min_t2c) / (max_t2c - min_t2c)
                df['{}_scaled_t2c'.format(agent)] = scaled_t2c
                t2c_ratio = metric_fn(inputs, mode='ratio')
                df['{}_t2c_ratio'.format(agent)] = t2c_ratio
            elif metric_name == 'entropy':
                inputs = np.stack(df['{}_probabilities'.format(agent)].to_numpy(), axis=0)
                metric = metric_fn(inputs, axis=1)
                df['{}_{}'.format(agent, metric_name)] = metric
            elif metric_name == 'top_confidence':
                df = add_confidence(
                    df=df,
                    agent=agent,
                    top=True)
    return df

def add_alignment(df):
    agent_pairs = [
        ('human', 'explainer'),
        ('human', 'model'),
        ('model', 'explainer')]
    for agent1, agent2 in agent_pairs:
        agent1_predictions = df['{}_predictions'.format(agent1)]
        agent2_predictions = df['{}_predictions'.format(agent2)]
        alignment = np.where(agent1_predictions == agent2_predictions, 1, 0)
        df['{}_{}_alignment'.format(agent1, agent2)] = alignment

    return df

def add_5_way_breakdown(df):
    cases = [
        ('case1', lambda h, e, m: (m == e) & (e == h)),  # model = explainer = human
        ('case2', lambda h, e, m: (m == e) & (e != h)),  # model = explainer != human
        ('case3', lambda h, e, m: (m == h) & (e != h)),  # model = human != explainer
        ('case4', lambda h, e, m: (e == h) & (m != h)),  # explainer = human != model
        ('case5', lambda h, e, m: (m != e) & (e != h) & (m != h)),  # model != explainer != human != model
    ]
    running_sum = 0
    for name, lambda_fn in cases:
        case_col = list(map(lambda_fn, df['human_predictions'], df['explainer_predictions'], df['model_predictions']))
        df[name] = case_col
        running_sum += df[name].sum()
        print("{:.3f}% samples in {}".format(100 * df[name].sum() / len(df), name))
    assert running_sum == len(df)

    return df

def build_csv(outputs, verbose=True):
    df = pd.DataFrame()
    if verbose:
        print("Adding outputs to dataframe...")
    df = add_outputs(
        df=df,
        outputs=outputs
    )

    if verbose:
        print("Adding uncertainty to dataframe...")
    df = add_uncertainty(df=df)

    if verbose:
        print("Adding alignment to dataframe...")
    df = add_alignment(df=df)

    if verbose:
        print("Adding 5 way breakdown to dataframe...")
    df = add_5_way_breakdown(df=df)

    # Add image indexes
    image_idxs = df.index.values
    df['image_idxs'] = image_idxs

    return df


def generate_csv(dataset_name,
                 human_output_path,
                 model_output_path,
                 explainer_output_path,
                 csv_save_dir,
                 verbose=True):

    # Check if file already exists at csv_save_path
    csv_save_path = os.path.join(csv_save_dir, 'uncertainties_alignment.csv')
    if os.path.exists(csv_save_path):
        print("File exists at {}. Aborting".format(csv_save_path))
        return
    ensure_dir(csv_save_dir)

    # Check dataset is valid
    assert dataset_name in DATASETS_AVAILABLE, "Unsupported dataset '{}'. Try one of {}".format(
        dataset_name, DATASETS_AVAILABLE)

    # Get outputs (output aka logits/probability/prediction) from all agents
    outputs = get_outputs(
        dataset_name=dataset_name,
        human_output_path=human_output_path,
        model_output_path=model_output_path,
        explainer_output_path=explainer_output_path
    )
    print(outputs.keys())
    df = build_csv(outputs)
    df.to_csv(csv_save_path)

    if verbose:
        print("Saved dataframe to {}".format(csv_save_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True, help='Name of dataset')
    parser.add_argument('--human_output_path', type=str, required=True, help='Path the human probabilities')
    parser.add_argument('--model_output_path', type=str, required=True, help='Path the model outputs/probabilities')
    parser.add_argument('--explainer_output_path', type=str, required=True, help='Path the explainer outputs/probabilities')
    parser.add_argument('--csv_save_dir', type=str, required=True, help='Path to save CSV to')

    args = parser.parse_args()

    generate_csv(
        dataset_name=args.dataset_name,
        human_output_path=args.human_output_path,
        model_output_path=args.model_output_path,
        explainer_output_path=args.explainer_output_path,
        csv_save_dir=args.csv_save_dir,
        verbose=True)
