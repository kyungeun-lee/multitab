from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, r2_score, log_loss
import torch, optuna, os
from scipy.special import expit, softmax
import numpy as np

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_auroc(y_true, y_pred):
    if (y_true.ndim == 1) & (y_pred.ndim == 2):
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return None

def calculate_multi_auroc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    except ValueError:
        return None

def calculate_f1_score(y_true, y_pred, average='binary'):
    return f1_score(y_true, y_pred, average=average)

def calculate_log_loss(y_true, y_pred, labels=None):
    try:
        return log_loss(y_true, y_pred, labels=labels)
    except ValueError:
        return None

def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse ** 0.5

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_metric(y_true, y_pred, y_prob, tasktype, datatype, prob=False):
    y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    y_pred = np.nan_to_num(y_pred)
    if tasktype == "regression":
        return {f'rmse_{datatype}': calculate_rmse(y_true, y_pred), f'r2_{datatype}': calculate_r2(y_true, y_pred)}
    elif tasktype == "binclass":
        if not prob:
            y_prob = expit(y_prob)
        return {f'acc_{datatype}': calculate_accuracy(y_true, y_pred), f'auroc_{datatype}': calculate_auroc(y_true, y_prob), 
                f'f1_{datatype}': calculate_f1_score(y_true, y_pred), f'logloss_{datatype}': calculate_log_loss(y_true, y_prob, labels=[0, 1])}
    elif tasktype == "multiclass":
        if not prob:
            y_prob = softmax(y_prob, axis=1)
        y_true_classes = np.argmax(y_true, axis=1) if len(y_pred.shape) == 1 else y_true
        return {f'acc_{datatype}': calculate_accuracy(y_true_classes, y_pred), f'auroc_{datatype}': calculate_multi_auroc(y_true, y_prob), 
                f'f1_{datatype}': calculate_f1_score(y_true_classes, y_pred, average='weighted'), f'logloss_{datatype}': calculate_log_loss(y_true, y_prob)}

def save_fig(study, savepath):
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig2 = optuna.visualization.plot_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator())
    fig3 = optuna.visualization.plot_slice(study)
    fig4 = optuna.visualization.plot_parallel_coordinate(study)

    fig1.write_image(os.path.join(savepath, 'optimization_history.png'))
    fig2.write_image(os.path.join(savepath, 'param_importance.png'))
    fig3.write_image(os.path.join(savepath, 'update_slice.png'))
    fig4.write_image(os.path.join(savepath, 'update_parallel.png'))


def log_training_options(fname, file_path="/home/lab-di/squads/supertab/multitab/results/optim_logs/error.log"):
    with open(file_path, "a") as file:
        file.write(f'Error raised. Unbalanced class problem. ({fname})\n')

def check_if_fname_exists_in_error(fname, file_path="/home/lab-di/squads/supertab/multitab/results/optim_logs/error.log"):
    result = True
    try:
        with open(file_path, "r") as file:
            for line in file:
                if fname in line:
                    result = False
    except FileNotFoundError:
        result = True
    return result

def is_study_todo(study, tasktype, optimal_value=1.0, num_trials=100):
    # Check if the study reached the optimal goal set in the callback
    if tasktype != "regression":
        if study.best_value >= optimal_value:
            print(f"Study reached the optimal value of {optimal_value}.")
            return False

    # Check if the study has completed the minimum number of trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == num_trials:
        print(f"Study has completed the required trials of {num_trials}.")
        return False

    donelen = len(completed_trials)
    print(f'Study is not yet complete. {donelen}')
    return True

