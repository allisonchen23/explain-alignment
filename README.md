# README

## Running Hyperparameter Searches

### Logging on wandb
SBATCH command: `sbatch bash/explainer_hparam_search/wandb_train_<explainer_type>_hparam.sh`. 
* This will pass in the appropriate config file from `configs/explainer_hparam_search`. The flag `--build_save_dir` will create a save directory hierarchy based on the explainer parameters from the config file including `dataset.args.input_feature_path` and `arch.args.n_hidden_features`.
* This will run the file `explainer_hparam_search_wandb.py` which sets up a wandb hyperparameter sweep and start agents for each learning rate and weight decay combination.
* The training script can be specified as an argument `--train_script_path`.

### Without logging on wandb
SBATCH command: `sbatch bash/explainer_hparam_search/train_<explainer_type>_hparam.sh`.

## Running Repeated Trials
SBATCH command: `sbatch --array [1-n_jobs]%<n_jobs_at_once> <path_to_bash_file> <config_json_name>`. For example, `sbatch --array [1-20]%5 bash/repeated_trials/run_trials.sh cifar_pixel_NA` will run 20 trials (at most 5 at a time) using the config file `configs/repeated_trials/cifar_pixel_NA.json`. 
