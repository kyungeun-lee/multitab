
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

# Initialize argument parser
parser = argparse.ArgumentParser()

# Add arguments to the parser for GPU ID, OpenML dataset ID, code directory, model name, preprocessing method, and categorical feature threshold
parser.add_argument("--gpu_id", type=int, default=4, help="gpu index")
parser.add_argument("--openml_id", type=int, default=4538, help="dataset index (See dataset_id.json for detailed information)")
parser.add_argument("--modelname", type=str, default='catboost', 
                    choices=['xgboost', 'catboost', 'lightgbm', 'mlp', 'ftt', 'resnet', 't2gformer'])
parser.add_argument("--preprocessing", type=str, default="quantile", 
                    choices=['standardization', 'quantile'], help="numerical feature preprocessing method")
parser.add_argument("--cat_threshold", type=int, default=20, help="categorical feature definition")
parser.add_argument("--savepath", type=str, default="results/optim_logs", help="path to save the results")

# Parse the arguments
args = parser.parse_args()

# Ensure that XGBoost and MLP have no special module for categorical features
if args.modelname in ['xgboost', 'mlp']:
    assert args.cat_threshold == 0
 
# Load dataset information from a JSON file
with open(f'dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info.get(str(args.openml_id))['tasktype']

# Define directory for saving logs and create it if it does not exist
if not args.savepath.endswith("optim_logs"):
    savepath = os.path.join(args.savepath, "optim_logs")
else:
    savepath = args.savepath
if not os.path.exists(savepath):
    os.makedirs(savepath)
fname = os.path.join(savepath, f'data={args.openml_id}..model={args.modelname}..numprep={args.preprocessing}..catprep={args.cat_threshold}.pkl')
    
# Prevent duplicated running by checking if the logs exist
train = True
print(fname)
if os.path.exists(fname):
    done_result = joblib.load(fname)
    print("Already done!", done_result.best_trial.user_attrs)
    train = False
    
# Main part starts here:
if train:
    # Set GPU environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_info = '{0}:{1}'.format(os.uname().nodename, args.gpu_id)
    
    # Load dataset with specified preprocessing
    quantile = bool(args.preprocessing == "quantile")
    dataset = TabularDataset(args.openml_id, tasktype, device=device, cat_threshold=args.cat_threshold, modelname=args.modelname, quantile=quantile)
    
    # Split dataset into training, validation, and test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std = dataset.y_std.item() if tasktype == "regression" else None
    
    # Check for class imbalance problems in multiclass tasks with specific models
    if (tasktype == "multiclass") & (args.modelname in ["catboost", "xgboost", "lightgbm"]):
        if y_train.size(1) != y_train.unique(dim=1).size(1):            
            raise ValueError # "Unknown class problem" --- Inherent challenges in GBDTs
    
    # Define optimization trials
    def objective(trial):
        print("### Start: ", trial.datetime_start.strftime("%m/%d %H:%M:%S"))
        params = get_search_space(trial, args.modelname)    
        
        model = getmodel(args.modelname, params, tasktype, dataset, args.openml_id, X_train.shape[1], y_train.shape[1], device)
        model.fit(X_train, y_train, X_val, y_val)
        
        preds_val = model.predict(X_val)
        preds_test = model.predict(X_test)
        
        if tasktype == "regression":
            val_metrics = calculate_metric(y_val*y_std, preds_val*y_std, tasktype, 'val')
            test_metrics = calculate_metric(y_test*y_std, preds_test*y_std, tasktype, 'test')
        else:
            val_metrics = calculate_metric(y_val, preds_val, tasktype, 'val')
            test_metrics = calculate_metric(y_test, preds_test, tasktype, 'test')
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

        # Optimization objectives: RMSE(val) for regression tasks, Accuracy(val) for classification tasks
        if tasktype == "regression":
            return val_metrics["rmse_val"]
        else:
            return val_metrics["acc_val"]

    # Run optimization with 100 trials without exception
    study = optuna.create_study(direction='minimize') if tasktype == "regression" else optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    # Save optimization history
    print("#############################################")
    print(env_info)
    print(study.best_trial.user_attrs)
    joblib.dump(study, fname)
    print(fname)
    print("#############################################")
    
#     save_fig(study, savepath) ### only for convenience
