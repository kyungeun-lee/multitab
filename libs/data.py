from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import openml, torch
import numpy as np
import pandas as pd
import sklearn.datasets
import scipy.stats
from sklearn.preprocessing import QuantileTransformer

def get_batch_size(n): 
    ### n = train data size
    if n > 50000:
        return 1024
    elif n > 10000:
        return 512
    elif n > 5000:
        return 256
    elif n > 1000:
        return 128
    else:
        return 64

def load_data(openml_id):
    if openml_id == 999999:
        dataset = sklearn.datasets.fetch_california_housing()
        X = pd.DataFrame(dataset['data'])
        y = pd.DataFrame(dataset['target'])
    elif openml_id == 43611:
        dataset = openml.datasets.get_dataset(openml_id)
        print(f'Dataset is loaded.. Data name: {dataset.name}, Target feature: class')
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target="class"
        )
    else:
        dataset = openml.datasets.get_dataset(openml_id)
        print(f'Dataset is loaded.. Data name: {dataset.name}, Target feature: {dataset.default_target_attribute}')
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
    
    if openml_id == 537:
        y = y / 10000
        
    print("RAW", X.shape)
    nan_counts = X.isna().values.sum()
    cell_counts = X.shape[0] * X.shape[1]
    n_samples = X.shape[0]
    n_cols = X.shape[1]
    print(nan_counts / cell_counts)
    ### Preprocess NaN (Section 3.2)
    # 1. Remove columns containing more than 50% NaN values
    nan_cols = X.isna().sum(0)
    valid_cols = nan_cols.loc[nan_cols < (0.5*len(X))].index.tolist()
    X = X[valid_cols]
    # 2. Exclude samples containing any NaN values in either inputs or labels
    nan_idx = X.isna().any(axis=1)
    X = X[~nan_idx].reset_index(drop=True)
    y = y[~nan_idx].reset_index(drop=True)
    
    # 3. Define categorical features
    for col in X.select_dtypes(exclude=['float', 'int']).columns:
        colencoder = LabelEncoder()
        X[col] = colencoder.fit_transform(X[col])
    
    y = y.values
    if isinstance(y[0], str):
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
    
    return X.values, y

def one_hot(y):
    num_classes = len(np.unique(y))
    min_class = y.min()
    enc = LabelEncoder()
    y_ = enc.fit_transform(y - min_class)
    return np.eye(num_classes)[y_]

def split_data(X, y, tasktype, seed=123456, device='cuda'):
    
    if tasktype == "multiclass":
        y = one_hot(y)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=0.5, random_state=seed)
    
    X_train = torch.from_numpy(X_train).type(torch.float32).to(device)
    X_val = torch.from_numpy(X_val).type(torch.float32).to(device)
    X_test = torch.from_numpy(X_test).type(torch.float32).to(device)

    y_train = torch.from_numpy(y_train).type(torch.float32).to(device)
    y_val = torch.from_numpy(y_val).type(torch.float32).to(device)
    y_test = torch.from_numpy(y_test).type(torch.float32).to(device)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def cat_num_features(X_train, cat_threshold=20):
    num_features = X_train.shape[1]
    
    counts = torch.tensor([X_train[:, i].unique().numel() for i in range(num_features)])
    if cat_threshold == None:
        X_cat = []
        X_num = np.arange(num_features)
    else:
        X_cat = np.where(counts <= cat_threshold)[0].astype(int)
        X_num = np.array([int(i) for i in range(num_features) if not i in X_cat])
    return (X_cat, counts[X_cat], X_num)

# Numerical feature preprocessing: Standardization
def standardization(X, X_mean, X_std, y, y_mean=0, y_std=1, num_indices=[], tasktype='multiclass'):
    X[:, num_indices] = (X[:, num_indices] - X_mean[num_indices]) / (X_std[num_indices] + 1e-10)
    if tasktype == "regression":
        y = (y - y_mean) / (y_std + 1e-10)
    return (X, y)

# Numerical feature preprocessing: Quantile transform
def quant(X_train, X_val, X_test, y_train, y_val, y_test, y_mean=0, y_std=1, num_indices=[], tasktype='multiclass'):
    device = X_train.get_device()
    quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train[:, num_indices] = torch.tensor(quantile_transformer.fit_transform(X_train[:, num_indices].cpu().numpy()), device=device)
    X_val[:, num_indices] = torch.tensor(quantile_transformer.transform(X_val[:, num_indices].cpu().numpy()), device=device)
    X_test[:, num_indices] = torch.tensor(quantile_transformer.transform(X_test[:, num_indices].cpu().numpy()), device=device)
    if tasktype == "regression":
        y_train = (y_train - y_mean) / (y_std + 1e-10)
        y_val = (y_val - y_mean) / (y_std + 1e-10)
        y_test = (y_test - y_mean) / (y_std + 1e-10)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, openml_id, tasktype, device, labeled_data=1,
                 cat_threshold=1e+10, seed=123456, modelname="xgboost", cat_ordered=False, normalize=True, quantile=False):
        X, y = load_data(openml_id)
            
        self.tasktype = tasktype
        
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = split_data(X, y, self.tasktype, seed=seed)
        (self.X_cat, self.X_categories, self.X_num) = cat_num_features(torch.tensor(X), cat_threshold=cat_threshold)
        if modelname in ["ftt", "resnet", "t2gformer", "catboost", "lightgbm"]:
            for cat_dim in self.X_cat:
                unique_values = torch.cat([self.X_train[:, cat_dim].unique(), self.X_val[:, cat_dim].unique(), self.X_test[:, cat_dim].unique()]).unique() 
                
                mapping = {v.item(): idx for (idx, v) in enumerate(unique_values)}
                self.X_train[:, cat_dim] = torch.tensor([mapping[v.item()] for v in self.X_train[:, cat_dim]])
                self.X_val[:, cat_dim] = torch.tensor([mapping[v.item()] for v in self.X_val[:, cat_dim]])
                self.X_test[:, cat_dim] = torch.tensor([mapping[v.item()] for v in self.X_test[:, cat_dim]])
        print("input dim: %i, cat: %i, num: %i" %(self.X_train.size(1), len(self.X_cat), len(self.X_num)))
        
        self.batch_size = get_batch_size(len(self.X_train))
        self.X_mean = self.X_train.mean(0)
        self.X_std = self.X_train.std(0)
        self.y_mean = self.y_train.type(torch.float).mean(0)
        self.y_std = self.y_train.type(torch.float).std(0)
                
        if quantile & (len(self.X_num) > 0) & normalize:
            (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = quant(
                self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test,
                self.y_mean, self.y_std, num_indices=self.X_num, tasktype=self.tasktype)
        elif (not quantile) & normalize:
            (self.X_train, self.y_train) = standardization(self.X_train, self.X_mean, self.X_std, self.y_train, self.y_mean, self.y_std, num_indices=self.X_num, tasktype=self.tasktype)
            (self.X_val, self.y_val) = standardization(self.X_val, self.X_mean, self.X_std, self.y_val, self.y_mean, self.y_std, num_indices=self.X_num, tasktype=self.tasktype)
            (self.X_test, self.y_test) = standardization(self.X_test, self.X_mean, self.X_std, self.y_test, self.y_mean, self.y_std, num_indices=self.X_num, tasktype=self.tasktype)
            
    def __len__(self, data):
        if data == "train":
            return len(self.X_train)
        elif data == "val":
            return len(self.X_val)
        else:
            return len(self.X_test)
    
    def _indv_dataset(self):
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)
    
    def __getitem__(self, idx, data):
        if data == "train":
            return self.X_train[idx], self.y_train[idx]
        elif data == "val":
            return self.X_val[idx], self.y_val[idx]
        else:
            return self.X_test[idx], self.y_test[idx]