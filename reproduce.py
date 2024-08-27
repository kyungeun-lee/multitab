
## Main file for reproducing performance with the optimal configuration for a given set of [algorithm, dataset, preprocessing method].
## Paper info: MultiTab: A Comprehensive Benchmark Suite with Multi-Dimensional Analysis in Tabular Domains
## Contact author: Kyungeun Lee (kyungeun.lee@lgresearch.ai)

import optuna, argparse, os, torch, json, joblib, time, datetime
from libs.data import TabularDataset
from libs.model import *
from libs.eval import *
from libs.search_space import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize argument parser
parser = argparse.ArgumentParser()

# Add arguments to the parser for GPU ID, OpenML dataset ID, code directory, model name, preprocessing method, and categorical feature threshold
parser.add_argument("--gpu_id", type=int, default=5)
parser.add_argument("--openml_id", type=int, default=25) #4538 458 // 25 45062 // 999999
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--modelname", type=str, default='tabpfn', 
                    choices=['lr', 'randomforest', 'xgboost', 'catboost', 'lightgbm', 'mlp', 'ftt', 'resnet', 't2gformer', 'tabpfn'])
parser.add_argument("--preprocessing", type=str, default="quantile", 
                    choices=['standardization', 'quantile'], help="numerical feature preprocessing method")
parser.add_argument("--cat_threshold", type=int, default=0, help="categorical feature definition")
parser.add_argument("--ensemble", type=int, default=0, help="ensemble trial") # optional argument for allowing the repeated reproduction
parser.add_argument("--savepath", type=str, default="results", help="path to save the results")

# Parse the arguments
args = parser.parse_args()
   
# Ensure that XGBoost and MLP have no special module for categorical features
if args.modelname in ['xgboost', 'mlp']:
    assert args.cat_threshold == 0

# Load dataset information from a JSON file
with open(f'dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info.get(str(args.openml_id))['tasktype']

# Define directory for loading the optimization logs and for saving reproducing logs
fname = os.path.join(args.savepath, f'optim_logs/seed={args.seed}/data={args.openml_id}..model={args.modelname}..numprep={args.preprocessing}..catprep={args.cat_threshold}.pkl')
print(fname)
if args.modelname not in ["tabpfn", "lr"]:
    assert os.path.exists(fname) # If there is no optimization log, assertion error will be raised
    
    # Load the optimization logs
    opt_logs = joblib.load(fname)

# Make directory for saving the results
if not os.path.exists(os.path.join(args.savepath, f'reproduce_logs/seed={args.seed}/{args.ensemble}')):
    os.makedirs(os.path.join(args.savepath, f'reproduce_logs/seed={args.seed}/{args.ensemble}'))
fname2 = os.path.join(args.savepath, f'reproduce_logs/seed={args.seed}/{args.ensemble}/data={args.openml_id}..model={args.modelname}..numprep={args.preprocessing}..catprep={args.cat_threshold}.npy')

# Main part starts here (Prevent the duplicated running):
if not os.path.exists(fname2):
    
    if args.modelname not in ["tabpfn", "lr"]:
        params = opt_logs.best_params
        # Add default(fixed) parameters
        params = add_default_params(args.modelname, params)
    
    # Set GPU environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_info = '{0}:{1}'.format(os.uname().nodename, args.gpu_id)

    # Load dataset with specified preprocessing
    quantile = bool(args.preprocessing == "quantile")
    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed, cat_threshold=args.cat_threshold, modelname=args.modelname, quantile=quantile)
    
    # Split dataset into training, validation, and test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std = dataset.y_std.item() if tasktype == "regression" else None

    # Check for class imbalance problems in multiclass tasks with specific models
    if (tasktype == "multiclass") & (args.modelname in ["catboost", "xgboost", "lightgbm"]):
        if y_train.size(1) != y_train.unique(dim=1).size(1):            
            raise ValueError # "Unknown class problem" --- Inherent challenges in GBDTs
    
    # Define and train the model
    output_dim = y_train.shape[1] if tasktype == "multiclass" else 1
    if args.modelname in ["tabpfn", "lr"]:
        model = getmodel(args.modelname, {}, tasktype, dataset, args.openml_id, X_train.shape[1], output_dim, device)
    else:
        model = getmodel(args.modelname, params, tasktype, dataset, args.openml_id, X_train.shape[1], output_dim, device)
    model.fit(X_train, y_train, X_val, y_val)
    
    # Model inference
    preds_test = model.predict(X_test)
    # For classification tasks with ensemble techniques, we should calculate probability or logits
    preds_test_prob = model.predict_proba(X_test) if tasktype != "regression" else None
        
    inference_results = {"Prediction": preds_test, "Probability": preds_test_prob}
    
    if tasktype == "regression":
        test_metrics = calculate_metric(y_test*y_std, preds_test*y_std, None, tasktype, 'test')
    else:
        test_metrics = calculate_metric(y_test, preds_test, preds_test_prob, tasktype, 'test')
    inference_results["Performance"] = test_metrics
    
    print(device, env_info, args.openml_id, data_info.get(str(args.openml_id))['name'], args.modelname, args.savepath)
    print(test_metrics)

    print("#############################################")
    print(env_info)
    np.save(fname2, inference_results)
    print("#############################################")
