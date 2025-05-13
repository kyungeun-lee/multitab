from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import openml, torch
import numpy as np
import pandas as pd
import sklearn.datasets
import scipy.stats
from sklearn.preprocessing import QuantileTransformer, StandardScaler

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
        categorical_indicator = []
        attribute_names = X.columns.tolist()
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
        
    # print("RAW", X.shape)
    nan_counts = X.isna().values.sum()
    cell_counts = X.shape[0] * X.shape[1]
    n_samples = X.shape[0]
    n_cols = X.shape[1]
    # print(nan_counts / cell_counts)
    ### Preprocess NaN (Section 3.2)
    # 1. Remove columns containing more than 50% NaN values
    nan_cols = X.isna().sum(0)
    valid_cols = nan_cols.loc[nan_cols < (0.5*len(X))].index.tolist()
    total_features = len(valid_cols)
    X = X[valid_cols]
    # 2. Exclude samples containing any NaN values in either inputs or labels
    nan_idx = X.isna().any(axis=1)
    X = X[~nan_idx].reset_index(drop=True)
    y = y[~nan_idx].reset_index(drop=True)
    
    # 3. convert categorical features into integers (but still they are categorical)
    cat_features = np.array(attribute_names)[categorical_indicator]
    cat_features = [c for c in cat_features if c in valid_cols]
    for v in valid_cols:
        if not v in cat_features:
            try:
                X[v].astype(np.float32)
            except ValueError:
                valid_cols.remove(v)
    X = X[valid_cols]

    cat_cols = [valid_cols.index(x) for x in cat_features]
    num_cols = [valid_cols.index(x) for x in valid_cols if not x in cat_features]
    cat_cardinality = [X[c].cat.categories.size for c in cat_features]
    for col in cat_features:
        colencoder = LabelEncoder()
        X[col] = colencoder.fit_transform(X[col])
    X = X.values
    for col in num_cols:
        if X[:, col].dtype == np.object_:
            X[:, col] = X[:, col].astype(np.float32)

    y = y.values
    if isinstance(y[0], str) or isinstance(y[0], (bool, np.bool_)):
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)

    print("full data size", X.shape)
    return X, y, cat_cols, cat_cardinality, num_cols

def one_hot(y):
    num_classes = len(np.unique(y))
    min_class = y.min()
    enc = LabelEncoder()
    y_ = enc.fit_transform(y - min_class)
    return np.eye(num_classes)[y_]

def split_data(X, y, tasktype, num_indices=[], seed=0, device='cuda'):
    
    if tasktype == "multiclass":
        y = one_hot(y)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_idx = list(kf.split(X))
    tr_idx, te_idx = fold_idx[seed]
    val_split_idx = (seed+1) % 10
    _, val_idx = fold_idx[val_split_idx]
    tr_idx = np.setdiff1d(tr_idx, val_idx)
        
    X_train = torch.from_numpy(X[tr_idx]).type(torch.float32).to(device)
    X_val = torch.from_numpy(X[val_idx]).type(torch.float32).to(device)
    X_test = torch.from_numpy(X[te_idx]).type(torch.float32).to(device)

    y_train = torch.from_numpy(y[tr_idx]).type(torch.float32).to(device)
    y_val = torch.from_numpy(y[val_idx]).type(torch.float32).to(device)
    y_test = torch.from_numpy(y[te_idx]).type(torch.float32).to(device)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std = prep_data(X_train, X_val, X_test, y_train, y_val, y_test, num_indices=num_indices, tasktype=tasktype)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std

## following Gorishniy et al., 2021
def prep_data(X_train, X_val, X_test, y_train, y_val, y_test, num_indices=[], tasktype='multiclass'):
    device = X_train.get_device()
    if len(num_indices) > 0:
        quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
        X_train[:, num_indices] = torch.tensor(quantile_transformer.fit_transform(X_train[:, num_indices].cpu().numpy()), device=device)
        X_val[:, num_indices] = torch.tensor(quantile_transformer.transform(X_val[:, num_indices].cpu().numpy()), device=device)
        X_test[:, num_indices] = torch.tensor(quantile_transformer.transform(X_test[:, num_indices].cpu().numpy()), device=device)
    if tasktype == "regression":
        standard_transformer = StandardScaler()
        y_train = torch.tensor(standard_transformer.fit_transform(y_train.reshape(-1, 1).cpu().numpy()).reshape(-1), device=device)
        y_std = standard_transformer.scale_.item()
        y_val = torch.tensor(standard_transformer.transform(y_val.reshape(-1, 1).cpu().numpy()).reshape(-1), device=device)
        y_test = torch.tensor(standard_transformer.transform(y_test.reshape(-1, 1).cpu().numpy()).reshape(-1), device=device)
    else:
        y_std = 1.
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, openml_id, tasktype, device, seed=1):
        
        X, y, self.X_cat, self.X_cat_cardinality, self.X_num = load_data(openml_id)
        self.tasktype = tasktype
        
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test), self.y_std = split_data(X, y, self.tasktype, num_indices=self.X_num, seed=seed, device=device)
        print("input dim: %i, cat: %i, num: %i" %(self.X_train.size(1), len(self.X_cat), len(self.X_num)))
        
        self.batch_size = get_batch_size(len(self.X_train))
        
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