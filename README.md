# README

## Running Hyperparameter Searches

### Logging on wandb
SBATCH command: `sbatch bash/explainer_hparam_search/wandb_hparam_search.sh <config_json_name>`. For example: `sbatch bash/explainer_hparam_search/wandb_hparam_search.sh train_ade20k_dense_ace_images_2000_concept_25_concept_explainer_hparam`. 
* This will pass in the appropriate config file from `configs/explainer_hparam_search`. The flag `--build_save_dir` will create a save directory hierarchy based on the explainer parameters from the config file including `dataset.args.input_feature_path` and `arch.args.n_hidden_features`.
* This will run the file `explainer_hparam_search_wandb.py` which sets up a wandb hyperparameter sweep and start agents for each learning rate and weight decay combination.
* The training script can be specified as an argument `--train_script_path`.

### Without logging on wandb
SBATCH command: `sbatch bash/explainer_hparam_search/train_<explainer_type>_hparam.sh`.

After running the hparam search, you can get the best trial directory by checking wandb.ai. Then, save the best results into its own directory with the script `save_best_hapram.py`. 

Example command to call: `python save_best_hparam.py --trial_dir saved/saved_explainers/ADE20K_dense_concept_explainer_hparam/ade20k/concepts/labeled/filter_attr_150/hidden_NA/0726_174556/trials/lr_0.5-wd_0`

## Running Repeated Trials
SBATCH command: `sbatch --array [1-n_jobs]%<n_jobs_at_once> <path_to_bash_file> <config_json_name>`. For example, `sbatch --array [1-20]%5 bash/repeated_trials/run_trials.sh cifar_pixel_NA` will run 20 trials (at most 5 at a time) using the config file `configs/repeated_trials/cifar_pixel_NA.json`. 

## Pipeline

### Saving model features & outputs/probabilities/predictions of images
* To save features: Use `src/save_features.py` Need to pass in config file such as the one found in `saved/places_model_ade20k_scene_labeled_features/0810_104502`
    * Default uses the resnet18 trained on Places365
* To save model outputs/probabilities/predictions: Run `src/save_model_outputs.py` which takes in path to a sklearn.LogisticRegression file and model features

### Get top labeled concepts as feature vectors
* Run `src/save_attributes.py` which will
    * Create one hot attributes for each split
