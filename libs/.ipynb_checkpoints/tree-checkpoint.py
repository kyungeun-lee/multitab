from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import torch
import pandas as pd
import numpy as np

class CatBoost(torch.nn.Module):
    def __init__(self, params, tasktype, cat_features=[]):
        loss_fn = {"multiclass": "MultiClass", "binclass": "CrossEntropy", "regression": "RMSE"}
        eval_fn = {"multiclass": "Accuracy", "binclass": "Accuracy", "regression": "RMSE"}
        model_fn = {"multiclass": CatBoostClassifier, "binclass": CatBoostClassifier, "regression": CatBoostRegressor}
            
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.model = model_fn[tasktype](loss_function=loss_fn[tasktype], eval_metric=eval_fn[tasktype], cat_features=cat_features, **params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0]), :]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train), :]
            y_train = y_train[~torch.isnan(y_train)]
        
        X_train = pd.DataFrame(X_train.cpu()).astype({k: 'int' for k in self.cat_features})
        y_train = np.argmax(y_train.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_train.cpu().numpy()
        
        X_val = pd.DataFrame(X_val.cpu()).astype({k: 'int' for k in self.cat_features})
        y_val = np.argmax(y_val.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_val.cpu().numpy()
         
        dtrain = Pool(X_train, label=y_train, cat_features=self.cat_features)
        dval = Pool(X_val, label=y_val, cat_features=self.cat_features)
        
        self.model.fit(dtrain, eval_set=dval, use_best_model=True, verbose=0)
        
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        return self.model.predict_proba(X_test)

    
class XGBoost(torch.nn.Module):
    def __init__(self, params, tasktype, cat_features=[]):
        loss_fn = {"multiclass": "multi:softmax", "binclass": "binary:logistic", "regression": "reg:squarederror"}
        model_fn = {"multiclass": XGBClassifier, "binclass": XGBClassifier, "regression": XGBRegressor}
            
        self.cat_features = cat_features
        self.tasktype = tasktype
        self.model = model_fn[tasktype](
            booster='gbtree', tree_method='gpu_hist', objective=loss_fn[tasktype], **params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0]), :]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train), :]
            y_train = y_train[~torch.isnan(y_train)]
        
#         import IPython; IPython.embed()
        X_train = pd.DataFrame(X_train.cpu()).astype({k: 'int' for k in self.cat_features})
        y_train = np.argmax(y_train.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_train.cpu().numpy()
        
        X_val = pd.DataFrame(X_val.cpu()).astype({k: 'int' for k in self.cat_features})
        y_val = np.argmax(y_val.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_val.cpu().numpy()
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=None, verbose=False)
        
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        return self.model.predict_proba(X_test)
    
    
class LightGBM(torch.nn.Module):
    def __init__(self, params, tasktype, cat_features=[]):
        loss_fn = {"multiclass": "multiclass", "binclass": "binary", "regression": "regression"}
        model_fn = {"multiclass": LGBMClassifier, "binclass": LGBMClassifier, "regression": LGBMRegressor}
            
        self.cat_features = cat_features
        self.tasktype = tasktype
        self.model = model_fn[tasktype](objective=loss_fn[tasktype], **params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0]), :]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train), :]
            y_train = y_train[~torch.isnan(y_train)]
        
        X_train = pd.DataFrame(X_train.cpu()).astype({k: 'category' for k in self.cat_features})
        y_train = np.argmax(y_train.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_train.cpu().numpy()
        
        X_val = pd.DataFrame(X_val.cpu()).astype({k: 'category' for k in self.cat_features})
        y_val = np.argmax(y_val.cpu().numpy(), axis=1) if self.tasktype == "multiclass" else y_val.cpu().numpy()
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=None, verbose=False, categorical_feature="auto")
        
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'category' for k in self.cat_features})
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'category' for k in self.cat_features})
        return self.model.predict_proba(X_test)