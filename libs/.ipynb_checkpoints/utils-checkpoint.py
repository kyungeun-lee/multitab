import logging
from tqdm import tqdm
import time
import scipy.stats

from libs.data import *
from libs.transform import BinShuffling
from typing import Any, Dict
from collections import defaultdict, OrderedDict
from scipy.io import arff
import numpy as np
import pandas as pd
import os, torchvision, torch
import sklearn.model_selection
import sklearn.datasets
import torch.nn.functional as F
import openml


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

def Binning(dataset, num_bins, device, binning_reg=True):
    
    num_bins = torch.FloatTensor(num_bins)
    binned_dataset = {"X_train": []}
    trainranks = dataset['X_train_ranks']
    rinterval = (trainranks.size(0) / num_bins).type(torch.int)
    binned_dataset["bin_interval"] = rinterval.to(device)
    rinterval = torch.tile(rinterval, (trainranks.size(0), 1))
    binned_dataset["X_train"] = (trainranks // rinterval)
    idx = torch.where(binned_dataset["X_train"] == num_bins)
    for (x, y) in zip(idx[0], idx[1]):
        binned_dataset["X_train"][x, y] = (num_bins[y] - 1)
    
    if binning_reg: ## Do standardization to bin indices
        binned_dataset["X_train"] = binned_dataset["X_train"].type(torch.float32)
        vmean = binned_dataset['X_train'].mean(0, keepdim=True)[0]
        vstd = binned_dataset['X_train'].std(0, keepdim=True)[0]
        binned_dataset["X_train"] = (binned_dataset["X_train"] - vmean) / (vstd+1e-10)

    return binned_dataset

####

def get_small_dataset(dataset, labeled_dataset):
    
    num_classes, counts = dataset['y_train'].unique(dim=0, sorted=True, return_counts=True)
    num_classes = num_classes.size(0)
    if (dataset['tasktype'] != 'regression') & (labeled_dataset < 1):
        num_samples_per_class = int(dataset['X_train'].size(0)*labeled_dataset // num_classes)
        assert num_samples_per_class > 0
        
        if (counts < num_samples_per_class).sum() > 0:
            ## ignore class balance
            np.random.seed(1024)
            unlabel_idx = np.random.choice(len(dataset['y_train']), int(len(dataset['y_train']) * (1-labeled_dataset)), replace=False)
            dataset['y_train'][unlabel_idx] = np.nan
        else:
            np.random.seed(1024)
            for c in range(num_classes):
                if num_classes == 2:
                    unlabel_idx = np.random.choice(
                        torch.where(dataset['y_train'] == c)[0].cpu().numpy(), len(torch.where(dataset['y_train'] == c)[0]) - num_samples_per_class, replace=False)
                else:
                    unlabel_idx = np.random.choice(
                        torch.where(dataset['y_train'][:, c].cpu() == 1)[0].cpu().numpy(), len(np.where(dataset['y_train'][:, c].cpu() == 1)[0]) - num_samples_per_class, replace=False)

                dataset['y_train'][unlabel_idx] = np.nan
    
    elif labeled_dataset < 1:
        np.random.seed(1024)
        unlabel_idx = np.random.choice(len(dataset['y_train']), int(len(dataset['y_train']) * (1-labeled_dataset)), replace=False)
        dataset['y_train'][unlabel_idx] = np.nan
    
    dataset['label_flag'] = torch.where(~torch.isnan(dataset['y_train']))[0].unique()
    
    return dataset
    
    
def weak_aug(dataset, init_bins, device, mask_alpha):
    
    num_features = len(dataset['columns'])
    if isinstance(init_bins, int) or isinstance(init_bins, float):
        init_bins = [int(init_bins)] * num_features
    elif isinstance(init_bins, torch.Tensor):
        init_bins = init_bins.tolist()
    
    init_binned_dataset = Binning(dataset, num_bins=init_bins, device=device)
    
    return init_binned_dataset, BinShuffling(mask_alpha, bin_interval=init_binned_dataset["bin_interval"], num_bins=init_bins)
    

def strong_aug(dataset, init_bins, device, mask_alpha,
               k_subsets, alpha, bmin=1):
    
    num_features = len(dataset['columns'])
    if isinstance(init_bins, int) or isinstance(init_bins, float):
        init_bins = [int(init_bins)] * num_features
    elif isinstance(init_bins, torch.Tensor):
        init_bins = init_bins.tolist()
    
    subset_vector = torch.empty(k_subsets, num_features).uniform_(0, 1) 
    ##other distribution okay
    subset_vector = torch.bernoulli(subset_vector) 
    ## for now, do not consider the duplicated subsets

    subset_vector = subset_vector * (1-alpha) + alpha # 1->1, 0->alpha

    old_bins = torch.Tensor(init_bins)
    new_bins = (old_bins * subset_vector).type(torch.int)
    if (new_bins < bmin).sum() > 0:
        new_bins[torch.where(new_bins < bmin)] = bmin

    strong_transform = []
    for subset in range(k_subsets):
        num_bins = new_bins[subset].tolist()
        binned = Binning(dataset, num_bins=num_bins, device=device, binning_reg=None)
        strong_transform.append(BinShuffling(mask_alpha, binned["bin_interval"], num_bins))
    
    return strong_transform, new_bins


