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
def build_csv(outputs):
    df = pd.DataFrame()
    df = add_outputs(
        df=df,
        outputs=outputs
    )
    df = add_uncertainty(df=df)


def generate_csv(dataset_name,
                 human_output_path,
                 model_output_path,
                 explainer_output_path):


    # Get outputs (output aka logits/probability/prediction) from all agents
    outputs = get_outputs(
        dataset_name=dataset_name,
        human_output_path=human_output_path,
        model_output_path=model_output_path,
        explainer_output_path=explainer_output_path
    )

    df = build_csv(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--human_output_path', type=str, required=True, help='Path the human probabilities')
    parser.add_argument('--model_output_path', type=str, required=True, help='Path the model outputs/probabilities')
    parser.add_argument('--explainer_output_path', type=str, required=True, help='Path the explainer outputs/probabilities')
