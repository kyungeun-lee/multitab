from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

class LR(torch.nn.Module):
    def __init__(self, tasktype):
        self.tasktype = tasktype
        self.model = LinearRegression() if self.tasktype == "regression" else LogisticRegression()
        
    def fit(self, X_train, y_train, X_val, y_val):
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        if self.tasktype == "multiclass":
            y_train = np.argmax(y_train, axis=1)
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())
    
    def predict_proba(self, X_test, logit=False):
        probs = self.model.predict_proba(X_test.cpu().numpy())
        if logit:
            # Prevent division by zero in logit calculation
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            if probs.shape[1] == 2:  # Binary classification
                logits = np.log(probs[:, 1] / (1 - probs[:, 1]))
                return logits
            else:  # Multiclass classification
                logits = np.log(probs / (1 - probs))
                return logits
        else:
            return probs

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class KNN(torch.nn.Module):
    def __init__(self, tasktype, n_neighbors=5):
        super(KNN, self).__init__()
        self.tasktype = tasktype
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors) if self.tasktype == "regression" else KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        if self.tasktype == "multiclass":
            y_train = np.argmax(y_train, axis=1)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())

    def predict_proba(self, X_test, logit=False):
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_test.cpu().numpy())
            if logit:
                # Prevent division by zero in logit calculation
                probs = np.clip(probs, 1e-9, 1 - 1e-9)
                if probs.shape[1] == 2:  # Binary classification
                    logits = np.log(probs[:, 1] / (1 - probs[:, 1]))
                    return logits
                else:  # Multiclass classification
                    logits = np.log(probs / (1 - probs))
                    return logits
            else:
                return probs
        else:
            raise NotImplementedError("Probability predictions are not supported for regression tasks.")

class DecisionTree(torch.nn.Module):
    def __init__(self, tasktype):
        super(DecisionTree, self).__init__()
        self.tasktype = tasktype
        self.model = DecisionTreeRegressor() if self.tasktype == "regression" else DecisionTreeClassifier()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        if self.tasktype == "multiclass":
            y_train = np.argmax(y_train, axis=1)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())

    def predict_proba(self, X_test, logit=False):
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_test.cpu().numpy())
            if logit:
                # Prevent division by zero in logit calculation
                probs = np.clip(probs, 1e-9, 1 - 1e-9)
                if probs.shape[1] == 2:  # Binary classification
                    logits = np.log(probs[:, 1] / (1 - probs[:, 1]))
                    return logits
                else:  # Multiclass classification
                    logits = np.log(probs / (1 - probs))
                    return logits
            else:
                return probs
        else:
            raise NotImplementedError("Probability predictions are not supported for regression tasks.")


class RandomForest(torch.nn.Module):
    def __init__(self, params, tasktype):
        self.params = params
        self.tasktype = tasktype
        if self.tasktype == "regression":
            self.model = RandomForestRegressor(**params)
        else:
            self.model = RandomForestClassifier(**params)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if self.tasktype == "multiclass":
            y_train = torch.argmax(y_train, dim=1)
        self.model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    
    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())
    
    def predict_proba(self, X_test, logit=False):
        probs = self.model.predict_proba(X_test.cpu().numpy())
        if logit:
            # Prevent division by zero in logit calculation
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            if probs.shape[1] == 2:  # Binary classification
                logits = np.log(probs[:, 1] / (1 - probs[:, 1]))
                return logits
            else:  # Multiclass classification
                logits = np.log(probs / (1 - probs))
                return logits
        else:
            return probs
    
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
    
    def predict_proba(self, X_test, logit=False):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        if logit:
            return self.model.predict(X_test, prediction_type='RawFormulaVal')
        else:
            return self.model.predict_proba(X_test)

    
class XGBoost(torch.nn.Module):
    def __init__(self, params, tasktype, cat_features=[]):
        loss_fn = {"multiclass": "multi:softmax", "binclass": "binary:logistic", "regression": "reg:squarederror"}
        model_fn = {"multiclass": XGBClassifier, "binclass": XGBClassifier, "regression": XGBRegressor}
            
        self.cat_features = cat_features
        self.tasktype = tasktype
        self.model = model_fn[tasktype](
            booster='gbtree', tree_method='hist', objective=loss_fn[tasktype], **params)
        
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
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test, logit=False):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'int' for k in self.cat_features})
        if logit:
            return self.model.predict(X_test, output_margin=True)
        else:
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
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=None, categorical_feature="auto")
        
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'category' for k in self.cat_features})
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test, logit=False):
        X_test = pd.DataFrame(X_test.cpu()).astype({k: 'category' for k in self.cat_features})
        if logit:
            return self.model.predict(X_test, raw_score=True)
        else:
            return self.model.predict_proba(X_test)