import numpy as np
import sklearn
from sklearn import inspection
from scipy import stats
import sys, os

sys.path.insert(0, 'src')
from utils.attribute_utils import hyperparam_search
import model.metric as module_metric
def top_2_confusion(soft_labels):
    '''
    Given soft label distribution, calculate difference between top 2 labels

    Arg(s):
        soft_labels : N x C np.array
            soft label array for N samples and C class predictions

    Returns:
        confusion : N-dim np.array
            confusion for each sample
    '''
    # Sort soft labels ascending
    sorted_soft_labels = np.sort(soft_labels, axis=-1)
    # Calculate difference of p(x) for top 2 classes
    top_2_difference = sorted_soft_labels[:, -1] - sorted_soft_labels[:, -2]
    # Confusion = 1 - difference (higher is worse)
    top_2_confusion = 1 - top_2_difference

    return top_2_confusion

def add_confidence(df, 
                   agent, 
                   top=True):
    column_name = '{}_probabilities'.format(agent)
    assert column_name in df.columns
    
    # Convert str -> numpy if necessary
    if type(df[column_name][0]) == str:
        df = convert_string_columns(df, [column_name])
    
    # Calculate confidence scores and add to DF
    probabilities = np.stack(df[column_name].to_numpy(), axis=0)
    if top:
        confidence = np.amax(probabilities, axis=1)
        df['{}_top_confidence'.format(agent)] = confidence
    else:  # confidence of bottom logit
        confidence = np.amin(probabilities, axis=1)
        df['{}_bottom_confidence'.format(agent)] = confidence
    return df

def sort_and_bin_df(df, sort_columns, n_bins):
    n_per_bin = len(df) // n_bins
    n_extra = len(df) % n_bins
    sorted_df = df.sort_values(sort_columns, ascending=True, ignore_index=True)

    # DV: proportion of images where the human and explainer disagree
    temp_df = df.copy()
    bin_rows = []
    iv_means = []
    iv_stds = []
    iv_ses = []
    start_idx = 0
    for bin_idx in range(n_bins):
        if bin_idx < n_extra:
            end_idx = start_idx + n_per_bin + 1
        else:
            end_idx = start_idx + n_per_bin
        cur_rows = sorted_df.iloc[start_idx:end_idx]

        bin_rows.append(cur_rows)
        iv_means.append(cur_rows[sort_columns[0]].mean())
        std = cur_rows[sort_columns[0]].std()
        se = std / np.sqrt(end_idx - start_idx)
        iv_stds.append(std)
        iv_ses.append(se)
        start_idx = end_idx
    return bin_rows, iv_means, iv_stds, iv_ses

def calculate_bin_disagreement(bin_rows,
                               agent1,
                               agent2):
    agents = ['human', 'explainer', 'model']
    assert agent1 in agents
    assert agent2 in agents
    
    disagreements = []
    for rows in bin_rows:
        assert '{}_predictions'.format(agent1) in rows.columns and \
            '{}_predictions'.format(agent2) in rows.columns
        agent1_preds = rows['{}_predictions'.format(agent1)].to_numpy()
        agent2_preds = rows['{}_predictions'.format(agent2)].to_numpy()
        disagreement = np.count_nonzero(agent1_preds != agent2_preds) / len(rows)
        disagreements.append(disagreement)
    return disagreements

def calculate_bin_agreement(bin_rows,
                               agent1,
                               agent2):
    agents = ['human', 'explainer', 'model']
    assert agent1 in agents
    assert agent2 in agents
    
    agreements = []
    n_samples = []
    for rows in bin_rows:
        assert '{}_predictions'.format(agent1) in rows.columns and \
            '{}_predictions'.format(agent2) in rows.columns
        agent1_preds = rows['{}_predictions'.format(agent1)].to_numpy()
        agent2_preds = rows['{}_predictions'.format(agent2)].to_numpy()
        agreement = np.count_nonzero(agent1_preds == agent2_preds) / len(rows)
        agreements.append(agreement)
        
        n_samples.append(len(rows))
    
    agreements = np.array(agreements)
    n_samples = np.array(n_samples)
    agreement_stds = np.sqrt(agreements * (1 - agreements))
    agreement_ses = agreement_stds / np.sqrt(n_samples)
    return agreements, agreement_stds, agreement_ses

def get_coefficient_significance(X,
                                 y,
                                 classifier):
    sse = np.sum(np.expand_dims((classifier.predict(X) - y) ** 2, axis=1), axis=0) / float(X.shape[0] - X.shape[1])
    se = np.array([
        np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                for i in range(sse.shape[0])
                ])

    t = classifier.coef_ / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), y.shape[0] - X.shape[1]))
    return t, p


def run_feature_importance_trial(train_rows,
                                 val_rows,
                                 x_names,
                                 y_names,
                                 metric_names,
                                 trial_id,
                                 logistic_regression_args,
                                 seed):
    # Get metric fns from names
    metric_fns = [getattr(module_metric, metric_name) for metric_name in metric_names]

    # Create structure for storing trial run
    cur_data = {}
    
    # Obtain x, y data
    train_x = train_rows[x_names].to_numpy()
    train_y = np.squeeze(train_rows[y_names].to_numpy())
    assert len(train_x) == len(train_y)
    val_x = val_rows[x_names].to_numpy()
    val_y = np.squeeze(val_rows[y_names].to_numpy())
    assert len(val_x) == len(val_y)

    # add iv, dv to data
    cur_data['iv'] = x_names
    cur_data['dv'] = y_names
    trial_key = "{} {}".format(trial_id, str((x_names, y_names)))

    # Scale the features to be between [0, 1]
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)

    # Perform hyperparameter search on classifier
    clf = hyperparam_search(
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_labels=val_y,
        scaler=None,
        logistic_regression_args=logistic_regression_args)

    print("Learned classifier to predict {} from {}".format(y_names, x_names))
    print("Coefficients: {}".format(clf.coef_))
    # Obtain predictions on validation set
    val_predictions = clf.predict(val_x)

    # Calculate metrics
    metrics = module_metric.compute_metrics(
        metric_fns=metric_fns,
        prediction=val_predictions,
        target=val_y,
        save_mean=True)
    for metric_name in metric_names:
        if '{}_mean'.format(metric_name) in metrics:
            cur_data[metric_name] = metrics['{}_mean'.format(metric_name)]
        elif metric_name in metrics:
            cur_data[metric_name] = metrics[metric_name]

    # Calculate variable sensitivity to each metric
    importance_results = inspection.permutation_importance(
        estimator=clf,
        X=train_x,
        y=train_y,
        scoring=metric_names,
        random_state=seed,
        n_repeats=50)
    for metric_name, cur_importance_results in importance_results.items():
        cur_data['{}_sensitivity_mean'.format(metric_name)] = cur_importance_results['importances_mean'].tolist()
        cur_data['{}_sensitivity_std'.format(metric_name)] = cur_importance_results['importances_std'].tolist()
    
    # Calculate coefficient importance 
    coefficient_t, coefficient_p = get_coefficient_significance(
        X=train_x,
        y=train_y,
        classifier=clf)
    cur_data['coefficient_t'] = coefficient_t.tolist()[0]
    cur_data['coefficient_p'] = coefficient_p.tolist()[0]
    
    return cur_data, trial_key