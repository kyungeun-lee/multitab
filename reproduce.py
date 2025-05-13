
## Main file for reproducing performance with the optimal configuration for a given set of [algorithm, dataset, preprocessing method].
## Paper info: MultiTab: A Comprehensive Benchmark Suite with Multi-Dimensional Analysis in Tabular Domains
## Contact author: Kyungeun Lee (kyungeun.lee@lgresearch.ai)

import optuna, argparse, os, torch, json, joblib, time, datetime, sys, shutil
from libs.data import TabularDataset
from libs.model import *
from libs.eval import *
from libs.search_space import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def is_study_todo(study, tasktype, optimal_value=1.0, num_trials=100):
    # Check if the study reached the optimal goal set in the callback
    if tasktype != "regression":
        if study.best_value >= optimal_value:
            # print(f"Study reached the optimal value of {optimal_value}.")
            return False

    # Check if the study has completed the minimum number of trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == num_trials:
        return False

    donelen = len(completed_trials)
    # print(donelen)
    return True

error_fname = "error.log"
errors = pd.DataFrame(columns=("seed", "data", "model"))
i = 0
with open(error_fname, "r") as file:
    for line in file:
        l = line.split("optim_logs/")[-1]
        seed = l.split("seed=")[-1].split("/")[0]
        data = l.split("data=")[-1].split("..")[0]
        model = l.split("model=")[-1].split(".pkl")[0]
        errors.loc[i] = [seed, data, model]
        i += 1

# Initialize argument parser
parser = argparse.ArgumentParser()

# Add arguments to the parser for GPU ID, OpenML dataset ID, code directory, model name, preprocessing method, and categorical feature threshold
parser.add_argument("--gpu_id", type=int, default=4)
parser.add_argument("--openml_id", type=int, default=10)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--savepath", type=str, default=".", help="path to save the results")

# Parse the arguments
args = parser.parse_args()

# Load dataset information from a JSON file
with open(f'/home/multitab/dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info.get(str(args.openml_id))['tasktype']

directory = os.path.join(args.savepath, f'reproduce_logs/seed={args.seed}/data={args.openml_id}')
if not os.path.exists(directory):
    os.makedirs(directory)

models = ["lr", "randomforest", "xgboost", "catboost", "lightgbm", "mlp", "embedmlp", "mlpplr", "resnet", "ftt", "t2gformer", "saint", "tabpfn"]
(init_hp, deepens, hyperens)
opts = [(True, 0, 0), #no HPO
        (False, 0, 0), #tuned
        (False, 1, 0), (False, 2, 0), (False, 3, 0), (False, 4, 0), #deep ensemble
        (False, 0, 1), (False, 0, 2), (False, 0, 3), (False, 0, 4)] #hyper ensemble
opt_dict = {"lr": [opts[0]], "tabpfn": [opts[0]],
            "randomforest": opts[:2], "xgboost": opts[:2], "catboost": opts[:2], "lightgbm": opts[:2],
            "mlp": opts, "embedmlp": opts, "mlpplr": opts, "ftt": opts, "resnet": opts, "t2gformer": opts, "saint": opts}

# Set GPU environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
torch.cuda.set_device(args.gpu_id)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env_info = '{0}:{1}'.format(os.uname().nodename, args.gpu_id)

# Load dataset with specified preprocessing
dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)

# Split dataset into training, validation, and test sets
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
y_std = dataset.y_std

for m in models:
    for (init_hps, ensemble_deep, ensemble_hyper) in opt_dict[m]:
        fname = os.path.join(directory, f'model={m}..init_hps={init_hps}..deep={ensemble_deep}..hyper={ensemble_hyper}.npy')
        todo = (os.path.exists(fname) == False)
        print("##########################################")
        print(env_info) 
        print(fname)
        print(todo)
        print("##########################################")
        if todo:
            if (m == "tabpfn") & (X_train.size(0) > 3000):
                np.save(fname, "ValueError: Not implemented.")
                sys.exit()
            
            if m not in ["tabpfn", "lr"]:
                # Load the optimization logs
                try:
                    opt_logs = joblib.load(os.path.join(args.savepath, f'optim_logs/seed={args.seed}/data={args.openml_id}..model={m}.pkl'))
                    not_complete = is_study_todo(opt_logs, tasktype)
                    assert not_complete == False
                except FileNotFoundError:
                    if len(errors[(errors["seed"] == str(args.seed)) & (errors["model"] == m) & (errors["data"] == str(args.openml_id))]) > 0:
                        np.save(fname, "ValueError: Not implemented.")
                        continue
                    else:
                        print("Not Yet")
                        continue
                        
                if init_hps:
                    params = opt_logs.trials[0].params
                elif ensemble_hyper > 0:
                    completed_trials = [trial for trial in opt_logs.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                    if len(completed_trials) <= ensemble_hyper:
                        np.save(fname, "ValueError: Not implemented.")
                        continue
                    assert len(completed_trials) > ensemble_hyper
                    if tasktype == "regression":
                        sorted_trials = sorted(completed_trials, key=lambda x: x.value)
                    else:
                        sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)
                    params = sorted_trials[ensemble_hyper].params
                else:
                    params = opt_logs.best_params
                # Add default(fixed) parameters
                params = add_default_params(m, params, args.openml_id)

            params = rearrange_params(m, args.openml_id, params)
            
            # Check for class imbalance problems in multiclass tasks with specific models
            if (tasktype == "multiclass") & (m in ["catboost", "xgboost", "lightgbm"]):
                if y_train.size(1) != y_train.unique(dim=1).size(1):            
                    raise ValueError # "Unknown class problem" --- Inherent challenges in GBDTs
            
            # Define and train the model
            output_dim = y_train.shape[1] if tasktype == "multiclass" else 1
            if m in ["tabpfn", "lr"]:
                model = getmodel(m, {}, tasktype, dataset, args.openml_id, X_train.shape[1], output_dim, device)
            else:
                model = getmodel(m, params, tasktype, dataset, args.openml_id, X_train.shape[1], output_dim, device)
        
            st = time.time()
            try:
                model.fit(X_train, y_train, X_val, y_val)
                et = time.time()
            except ValueError:
                np.save(fname, "ValueError: Not implemented.")
                sys.exit()
        
            # Model inference
            preds_test = model.predict(X_test)
            # For classification tasks with ensemble techniques, we should calculate probability or logits
            preds_test_prob = model.predict_proba(X_test, logit=True) if tasktype != "regression" else None
            inference_results = {"Prediction": preds_test, "Probability": preds_test_prob, "time": et - st}
            
            if tasktype == "regression":
                test_metrics = calculate_metric(y_test*y_std, preds_test*y_std, None, tasktype, 'test')
            else:
                test_metrics = calculate_metric(y_test, preds_test, preds_test_prob, tasktype, 'test')
            inference_results["Performance"] = test_metrics

            # print(inference_results["Prediction"])
            print(device, env_info, args.openml_id, data_info.get(str(args.openml_id))['name'], m, args.savepath)
            print(test_metrics)
        
            print("#############################################")
            np.save(fname, inference_results)
            print("#############################################")