from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, r2_score
import torch, optuna, os
import numpy as np

### todotodo : add r2 for regression

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_auroc(y_true, y_pred):
    y_scores = 1 / (1 + np.exp(-y_pred))
    return roc_auc_score(y_true, y_scores)

def calculate_f1_score(y_true, y_pred, average='binary'):
    return f1_score(y_true, y_pred, average=average)

def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse ** 0.5

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_metric(y_true, y_pred, tasktype, datatype):
    y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    y_pred = np.nan_to_num(y_pred)
    if tasktype == "regression":
        return {f'rmse_{datatype}': calculate_rmse(y_true, y_pred), f'r2_{datatype}': calculate_r2(y_true, y_pred)}
    elif tasktype == "binclass":
        return {f'acc_{datatype}': calculate_accuracy(y_true, y_pred), f'auroc_{datatype}': calculate_auroc(y_true, y_pred), f'f1_{datatype}': calculate_f1_score(y_true, y_pred)}
    elif tasktype == "multiclass":
        y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim == 2:
            if y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
        return {f'acc_{datatype}': calculate_accuracy(y_true, y_pred), f'f1_{datatype}': calculate_f1_score(y_true, y_pred, average='weighted')}


def save_fig(study, savepath):
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig2 = optuna.visualization.plot_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator())
    fig3 = optuna.visualization.plot_slice(study)
    fig4 = optuna.visualization.plot_parallel_coordinate(study)

    fig1.write_image(os.path.join(savepath, 'optimization_history.png'))
    fig2.write_image(os.path.join(savepath, 'param_importance.png'))
    fig3.write_image(os.path.join(savepath, 'update_slice.png'))
    fig4.write_image(os.path.join(savepath, 'update_parallel.png'))