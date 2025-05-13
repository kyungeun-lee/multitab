
## Main file for optimizing each model for a specific [dataset, preprocessing method] setup.
## Paper info: MultiTab: A Comprehensive Benchmark Suite with Multi-Dimensional Analysis in Tabular Domains
## Contact author: Kyungeun Lee (kyungeun.lee@lgresearch.ai)

import optuna, argparse, os, torch, json, joblib, datetime
from libs.data import TabularDataset
from libs.eval import *
from libs.search_space import *
from libs.model import getmodel
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

os.chdir("/home/multitab")
# Initialize argument parser
parser = argparse.ArgumentParser()

# Add arguments to the parser for GPU ID, OpenML dataset ID, code directory, model name, preprocessing method, and categorical feature threshold
parser.add_argument("--gpu_id", type=int, default=4, help="gpu index")
parser.add_argument("--openml_id", type=int, default=40672, help="dataset index (See dataset_id.json for detailed information)")
parser.add_argument("--seed", type=int, default=1, help="seed for dataset split (cross-validation)")
parser.add_argument("--modelname", type=str, default="resnet", 
                    choices=['randomforest', 'xgboost', 'catboost', 'lightgbm', 'mlp', 'embedmlp', 'mlpplr', 'ftt', 'resnet', 't2gformer', 'saint', 'modernnca', 'tabr']) #lr, tabpfn not here -- only in reproduce.py
parser.add_argument("--savepath", type=str, default=".", help="path to save the results")

# Parse the arguments
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# Load dataset information from a JSON file
with open(f'dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info.get(str(args.openml_id))['tasktype']
print(tasktype)

# Define directory for saving logs and create it if it does not exist
if not args.savepath.endswith("optim_logs"):
    savepath = os.path.join(args.savepath, "optim_logs", f'seed={args.seed}')
else:
    savepath = args.savepath
if not os.path.exists(savepath):
    os.makedirs(savepath)
fname = os.path.join(savepath, f'data={args.openml_id}..model={args.modelname}.pkl')
    
# Prevent duplicated running by checking if the logs exist
train = True
if os.path.exists(fname):
    study = joblib.load(fname)
    train = is_study_todo(study, tasktype)
else:
    study = optuna.create_study(direction='minimize') if tasktype == "regression" else optuna.create_study(direction='maximize')
    initial_trial = suggest_initial_trial(args.modelname)
    study.enqueue_trial(initial_trial)
    train = check_if_fname_exists_in_error(fname)

completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
remaining_trials = max(0, 100 - completed_trials_count)

# Main part starts here:
if train:
    # Set GPU environment variable
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_info = '{0}:{1}'.format(os.uname().nodename, args.gpu_id)
    print(env_info, device)
    
    # Load dataset
    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)
    
    # Split dataset into training, validation, and test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std = dataset.y_std
    
    # Define optimization trials
    def objective(trial):
        print("### Start: ", trial.datetime_start.strftime("%m/%d %H:%M:%S"))
        params = get_search_space(trial, args.modelname, num_features=X_train.size(1), data_id=args.openml_id)
        
        output_dim = y_train.shape[1] if tasktype == "multiclass" else 1
        model = getmodel(args.modelname, params, tasktype, dataset, args.openml_id, X_train.shape[1], output_dim, device)
        model.fit(X_train, y_train, X_val, y_val)
        
        preds_val = model.predict(X_val); preds_test = model.predict(X_test)
        if tasktype == "regression":
            probs_val, probs_test = None, None
        else:
            probs_val = model.predict_proba(X_val); probs_test = model.predict_proba(X_test)

        if tasktype == "regression":
            val_metrics = calculate_metric(y_val*y_std, preds_val*y_std, probs_val, tasktype, 'val')
            test_metrics = calculate_metric(y_test*y_std, preds_test*y_std, probs_test, tasktype, 'test')
        else:
            val_metrics = calculate_metric(y_val, preds_val, probs_val, tasktype, 'val')
            test_metrics = calculate_metric(y_test, preds_test, probs_test, tasktype, 'test')
        for k, v in val_metrics.items():
            trial.set_user_attr(k, v)
        for k, v in test_metrics.items():
            trial.set_user_attr(k, v)
        
        print(device, env_info, args.openml_id, data_info.get(str(args.openml_id))['name'], args.modelname, savepath)
        print(val_metrics)
        print(test_metrics)
        now = datetime.datetime.now()
        duration = now - trial.datetime_start
        print(f'### Optimization time for trial {trial.number}: {duration.total_seconds():.0f} secs')
        
        trial.set_user_attr('training_time', duration.total_seconds())
        
        # Optimization objectives: RMSE(val) for regression tasks, Accuracy(val) for classification tasks
        if tasktype == "regression":
            return val_metrics["rmse_val"]
        else:
            return val_metrics["acc_val"]

    def stop_when_reached_optimal(study, trial):
        if study.best_value >= 1.0:
            study.stop()

    if tasktype == "regression":
        study.optimize(objective, n_trials=remaining_trials, callbacks=[lambda study, trial: joblib.dump(study, fname)])
    else:
        study.optimize(objective, n_trials=remaining_trials, callbacks=[stop_when_reached_optimal, lambda study, trial: joblib.dump(study, fname)])
    
    total_training_time = sum([trial.user_attrs['training_time'] for trial in study.trials])
    study.set_user_attr('total_training_time', total_training_time)
    
    # Save optimization history
    print("#############################################")
    print(env_info)
    print(study.best_trial.user_attrs)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(savepath, f'data={args.openml_id}..model={args.modelname}.csv'), index=False)
    joblib.dump(study, fname)
    print(fname)
    print("#######################################")