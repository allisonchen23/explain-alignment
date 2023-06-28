import numpy as np
import sklearn
from sklearn import inspection
from scipy import stats
import pandas as pd
import sys, os

sys.path.insert(0, 'src')
from utils.attribute_utils import hyperparam_search
from utils.visualizations import bar_graph, plot
from utils.utils import informal_log
import model.metric as module_metric


def top_2_confusion(soft_labels,
                    mode='difference'):
    '''
    Given soft label distribution, calculate difference between top 2 labels

    Arg(s):
        soft_labels : N x C np.array
            soft label array for N samples and C class predictions

    Returns:
        confusion : N-dim np.array
            confusion for each sample
    '''
    available_modes = ['ratio', 'difference']
    assert mode in available_modes, "mode must be one of {}".format(available_modes)
    # Sort soft labels ascending
    sorted_soft_labels = np.sort(soft_labels, axis=-1)
    # Calculate difference of p(x) for top 2 classes
    top_2_difference = sorted_soft_labels[:, -1] - sorted_soft_labels[:, -2]
    # Confusion = 1 - difference (higher is worse)
    if mode == 'difference':
        top_2_confusion = 1 - top_2_difference
    elif mode == 'ratio':
        top_logit = sorted_soft_labels[:, -1]
        top_2_confusion = top_2_difference / top_logit

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

# def calculate_bin_agreement(bin_rows,
#                                agent1,
#                                agent2):
#     agents = ['human', 'explainer', 'model']
#     assert agent1 in agents
#     assert agent2 in agents
    
#     agreements = []
#     n_samples = []
#     for rows in bin_rows:
#         assert '{}_predictions'.format(agent1) in rows.columns and \
#             '{}_predictions'.format(agent2) in rows.columns
#         agent1_preds = rows['{}_predictions'.format(agent1)].to_numpy()
#         agent2_preds = rows['{}_predictions'.format(agent2)].to_numpy()
#         agreement = np.count_nonzero(agent1_preds == agent2_preds) / len(rows)
#         agreements.append(agreement)
        
#         n_samples.append(len(rows))
    
#     agreements = np.array(agreements)
#     n_samples = np.array(n_samples)
#     agreement_stds = np.sqrt(agreements * (1 - agreements))
#     agreement_ses = agreement_stds / np.sqrt(n_samples)
#     return agreements, agreement_stds, agreement_ses

def calculate_bin_agreement(bin_rows,
                            column_name=None,
                            agent1=None,
                            agent2=None):
    '''
    
    Arg(s):
        bin_rows : list[pd.DataFrame]
            list of pandas dataframe representing rows to bin
        column_name : str or None
    '''
    agreements = []
    n_samples = []
    
    if column_name is not None:
        for rows in bin_rows:
            assert column_name in rows.columns
            agreement = rows[column_name]
            # check that it is all binary values
            unique_vals = set(pd.unique(agreement))
            assert {0} == unique_vals or {1} == unique_vals or {0, 1} == unique_vals, \
                "Expected binary values in agreement column. Received {}".format(unique_vals)
            
            agreements.append(np.count_nonzero(agreement) / len(rows))
            n_samples.append(len(rows))
            
    else:
        assert agent1 is not None and agent2 is not None, "If no column passed, must pass agent1 and agent2 in"
        agents = ['human', 'explainer', 'model']
        assert agent1 in agents
        assert agent2 in agents
        
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
    
    # Print count of unique predictions
    unique, unique_counts = np.unique(val_predictions, return_counts=True)
    print("Predictions:")
    for val, count in zip(unique, unique_counts):
        print("{}: {:.2f}".format(val, 100 * count / len(val_predictions)))
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
    
    return {
        'clf': clf,
        'cur_data': cur_data, 
        'trial_key': trial_key
    }

def correlated_variables(train_rows,
                         x_names):
    # Calculate Spearman correlation coefficient with all 9 variables
    corr = stats.spearmanr(train_rows[x_names].to_numpy())
    n_ivs = len(x_names)
    low_corrs = []
    med_corrs = []
    high_corrs = []
    
    for idx1 in range(n_ivs-1):
        for idx2 in range(idx1+1, n_ivs):
            var1 = x_names[idx1]
            var2 = x_names[idx2]

            cur_corr = corr.statistic[idx1][idx2]
            cur_p = corr.pvalue[idx1][idx2]
            cur_data = (var1, var2, cur_corr, cur_p)
            if np.abs(cur_corr) < 0.3:
                low_corrs.append(cur_data)
            elif np.abs(cur_corr) < 0.6:
                med_corrs.append(cur_data)
            else:
                high_corrs.append(cur_data)


    print("Low correlation pairs (<0.3)")
    low_corr_set = set()
    for row in low_corrs:
        print(row)
        low_corr_set.add((row[0], row[1]))

    print("\nModerate correlation pairs (0.3<= x < 0.6)")
    med_corr_set = set()
    for row in med_corrs:
        print(row)
        med_corr_set.add((row[0], row[1]))

    print("\nHigh correlation pairs (>=0.6)")
    high_corr_set = set()
    for row in high_corrs:
        print(row)
        high_corr_set.add((row[0], row[1]))

    return low_corr_set, med_corr_set, high_corr_set


def filter_df(df,
              ivs=None,
              dv=None):
    if ivs is not None:
        assert type(ivs) == list
        assert type(ivs[0]) == str
        for iv in ivs:
            df = df[df['iv'].str.contains(iv, na=False)]
    if dv is not None:
        assert type(dv) == str
        df = df[df['dv'].str.contains(dv)]
        
    return df

def string_to_list(string, 
                   dtype='string',
                   verbose=False):
    '''
    Given a string, convert it to a list of strings

    Arg(s):
        string : str
            string assumed in format of numbers separated by spaces, with '[' ']' on each end
        verbose : bool
            whether or not to print error messages
    Returns:
        list
    '''
    if type(string) != str:
        return string
    original_string = string

    if string[0] == '[':
        string = string[1:]
    if string[-1] == ']':
        string = string[:-1]

    list_string = string.split(", ")
    if dtype == 'str':
        for idx, s in enumerate(list_string):
            if s[0] == "'":
                list_string[idx] = s[1:]
            if s[-1] == "'":
                list_string[idx] = s[:-1]
    elif dtype == 'int' or dtype == 'float':
        list_string = [eval(i) for i in list_string]
    
    return list_string

def print_summary(df, 
                  metrics=['accuracy', 'precision', 'recall', 'f1', 'neg_log_loss'],
                  graph_metric=None,
                  print_coefficient_importance=False,
                  log_path=None):
    sensitivity_metrics = ['{}_sensitivity_mean'.format(metric) for metric in metrics]

    for idx, row in df.iterrows():
        informal_log("Row index {}".format(idx), log_path)
        informal_log("IV: {} \nDV: {}".format(row['iv'], row['dv']), log_path)
        informal_log("Metrics", log_path)
        ivs = string_to_list(row['iv'])
        
        # Print metrics and the variable sensitivity
        for metric in metrics:
            informal_log("\t{}: {:.4f}".format(metric, row[metric]), log_path)
            metric_sensitivities = string_to_list(
                row['{}_sensitivity_mean'.format(metric)],
                dtype='float')
            for iv, metric_sensitivity in zip(ivs, metric_sensitivities):
                informal_log("\t\t {} sensitivity: {:.4f}".format(iv, metric_sensitivity), log_path)
                
        # Print coefficient importance scores and p-values
        if print_coefficient_importance:
            informal_log("Coefficient significance:", log_path)
            coefficient_ts = string_to_list(row['coefficient_t'], dtype='float')
            coefficient_ps = string_to_list(row['coefficient_p'], dtype='float')
            for idx, iv in enumerate(ivs):
                coefficient_t = coefficient_ts[idx]
                coefficient_p = coefficient_ps[idx]

                informal_log("\t{} t: {:.4f} p-value: {:.4f}".format(iv, coefficient_t, coefficient_p), log_path)

    
def plot_metric_v_inputs(df, 
                         ivs,
                         dv,
                         graph_metric):
    assert graph_metric in df.columns
    labels = []
    metric_ys = []
        
    for idx, row in df.iterrows():
        ivs = string_to_list(row['iv'])
        
        iv_abbreviations = ''
        for iv in ivs:
            iv = iv.replace("'", "")
            iv = iv.split('_')
            for word in iv:
                iv_abbreviations += word[0]
            iv_abbreviations += '_'
        iv_abbreviations = iv_abbreviations[:-1]
        labels.append(iv_abbreviations)
        metric_ys.append(row[graph_metric])

    bar_graph(
        data=np.array([metric_ys]),
        labels=labels,
        xlabel='Source of IV',
        ylabel=graph_metric,
        title='{} for {}'.format(graph_metric, dv),
        xlabel_rotation=30)
    
def plot_alignment(df,
                   confusion_names,
                   n_bins,
                   fig=None,
                   ax=None,
                   alignment_col_name=None,
                   agent1=None,
                   agent2=None,
                   plot_xerr=True,
                   plot_yerr=True):
    xlabel = 'Scaled {} (n_bins={})'.format(confusion_names[0], n_bins)
    bin_rows, iv_means, iv_stds, iv_ses = sort_and_bin_df(
        df=df,
        sort_columns=confusion_names, 
        n_bins=n_bins)
    # Scale confusion/confidence to be between [0, 1]
    min_iv = np.amin(iv_means)
    max_iv = np.amax(iv_means)
    iv_means = (iv_means - min_iv) / (max_iv - min_iv)

    # Calculate agreement in each bin with STD and SEs
    cur_agreement, agreement_stds, agreement_ses = calculate_bin_agreement(
            bin_rows=bin_rows,
            column_name=alignment_col_name,
            agent1=agent1,
            agent2=agent2)
    # confusions.append(iv_means)
    # agreements.append(cur_agreement)

    # Calculate pearson's correlation
    pearsons = stats.pearsonr(
        x=iv_means,
        y=cur_agreement)
    
    slope, yint, r, p, std_err = stats.linregress(iv_means, cur_agreement)
    r_squared = r ** 2
    if alignment_col_name is not None:
        ylabel = '% samples of {}'.format(alignment_col_name)
        title = '{} vs {}'.format(alignment_col_name, confusion_names[0])
    else:
        ylabel = '% of samples of {}-{} alignment'.format(agent1, agent2)
        title = '{}-{} alignment vs {}'.format(agent1, agent2, confusion_names[0])
    fig, ax = plot(
        xs=[iv_means],
        ys=[cur_agreement],
        fig=fig,
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        ylimits=[0, 1.1],
        xlimits=[-0.1, 1.1],
        labels=['Pearsons: {:.3f} (p={:.6f})\ny={:.2f}x+{:.2f}; R-squared: {:.3f}'.format(
            pearsons.statistic, pearsons.pvalue, slope, yint, r_squared)],
        title=title,
        show=False)

    if plot_xerr:
        ax.errorbar(
            x=iv_means,
            y=cur_agreement,
            xerr=iv_stds,
            ecolor='black',
            capsize=3.0)
    if plot_yerr:
        ax.errorbar(
            x=iv_means,
            y=cur_agreement,
            yerr=agreement_ses,
            ecolor='black',
            capsize=3.0)
    
    return ax, iv_means, iv_stds, cur_agreement, agreement_ses, ylabel