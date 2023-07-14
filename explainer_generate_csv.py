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

DATASETS_AVAILABLE = ['cifar', 'ADE20K']


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
