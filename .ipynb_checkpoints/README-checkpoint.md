# MultiTab
The implementation codes of the paper "MultiTab: A Comprehensive Benchmark Suite for Multi-Dimensional Analysis in Tabular Domains"
(Paper url: TBU)

## Table of Contents
- [Overview](#overview)
- [Required libraries](#required libraries)
- [Dataset descriptions](#dataset descriptions)
- [Experiments](#experiments)
- [Examples](#examples)
- [License](#license)
- [Contact](#contact)

## Overview
`MultiTab` is a benchmark suite designed for multi-dimensional analysis in tabular domains. It includes 210 datasets spanning both classification and regression tasks. `MultiTab` categorizes the entire data collection into diverse sub-categories based on 10 specific characteristics of tabular data, such as data size, task types, sample irregularity, and the average cardinality of categorical features. For prediction algorithms, `MultiTab` incorporates seven model classes — three GBDTs and four NNs. These models are evaluated across five experimental setups, varying in numerical feature preprocessing methods, the definition of categorical features, and the application of ensembles. This results in a total of 28 algorithm-setup combinations. Comprehensive hyperparameter tuning is performed for each algorithm and dataset to determine the optimal configurations, consuming over 1132 GPU days. All optimization logs and results are available on this project page.

See our paper at [TBU].

## Required libraries
Our project has been built on Python 3.9. Here is the entire list of python libraries required (also available in `requirements.txt`):

``` swift
optuna==3.5.0
argparse==1.1
torch==2.0.1
joblib==1.2.0
sklearn==1.2.2
scipy==1.10.1
numpy==1.26.0
pandas==2.1.1
openml==0.14.1
tqdm==4.66.1
xgboost==1.7.5
catboost==1.2
lightgbm==3.3.5
```

## Dataset descriptions
Our benchmark suite encompasses a comprehensive collection of 210 datasets, all of which are publicly accessible through the OpenML python library. All datasets from OpenML are provided under the CC-BY license, which implies that the data is publicly available and has been shared with the appropriate consent and ethical considerations. OpenML ensures that datasets shared on their platform comply with their data-sharing guidelines, which include obtaining necessary consent where applicable.

Each dataset can be loaded by inserting the dataset IDs in 
``` swift
openml.datasets.get_dataset(DATASET_ID)
```
Due to its widespread use in numerous studies, we allow an exception for the CA dataset (ID: 999999) in the regression task, which we load from the sklearn python library instead of OpenML repository. Each dataset has exactly one train-validation-test split (8:1:1 with a fixed random seed), so all algorithms use the same splits of datasets.

Here is the complete list of 210 dataset used in this study:

> 25, 461, 466, 444, 23381, 51, 53, 337, 49, 59, 1063, 1467, 1510, 334, 29, 40981, 470, 1464, 37, 50, 31, 1479, 934, 1504, 1462, 42931, 41145, 1067, 44091, 1486, 41143, 3, 44160, 1471, 1489, 1487, 1043, 24, 44158, 1494, 40900, 44124, 44157, 45062, 151, 44126, 44131, 4534, 44125, 40536, 41147, 44122, 44089, 44090, 23512, 44162, 4135, 45060, 44123, 41162, 846, 40922, 44161, 41150, 45068, 44159, 45545, 42665, 44129, 1169, 551, 171, 7, 10, 48, 338, 4153, 452, 474, 1549, 455, 475, 5, 40496, 313, 42, 188, 11, 54, 458, 469, 1493, 1555, 307, 1492, 35, 454, 23, 1501, 14, 185, 20, 12, 16, 18, 22, 1466, 6, 46, 36, 43986, 372, 1497, 40499, 41027, 1475, 182, 40685, 1531, 1459, 40985, 26, 4538, 42345, 32, 43044, 41168, 45714, 1476, 1509, 42734, 41275, 45548, 40672, 41169, 1503, 1113, 41960, 150, 1567, 210, 497, 1099, 40916, 505, 560, 566, 524, 194, 511, 509, 456, 8, 42360, 531, 703, 534, 44968, 549, 43611, 45102, 43962, 4544, 41702, 41021, 41265, 43466, 43919, 315, 422, 503, 41700, 43926, 507, 44055, 44061, 42727, 344, 44136, 562, 189, 44056, 44145, 44062, 558, 44132, 42183, 44063, 44026, 537, 44054, 44057, 44984, 44059, 44068, 44137, 44133, 44963, 44134, 45012, 44148, 44066, 44064, 44069, 43928, 44005, 42370, 44027, 44065, 999999

The detailed description of the criteria of the MultiTab is available in Supplementary D of the paper. The list of datasets in each subcategory is available in the `.txt` file in `multitab` directory.
In addition, the metric for each criterion is available in `results/full_stats.csv` and `results/multitab.csv`.

### Experiments
We provide the implementations codes for (1) optimizing the best hyperparameter optimization using the Tree-structured Parzen Estimator provided by the optuna python library for 100 trials without exception; and (2) reproducing the best configuration found in (1).

#### 1. Optimization (`optimize.py`)
To ensure that each prediction algorithm reaches its optimal configuration, we conducted an extensive hyperparameter optimization across all datasets and algorithms, performing 100 trials without exception. The optimization objectives were tailored to the task types: validation accuracy for classification tasks and validation RMSE (Root Mean Squared Error) for regression tasks. The hyperparameter search space is described in Supplementary E of the paper and `libs/search_space.py`. As a result, the optimization logs will be saved at the predetermined directory. We provide the entire optimization logs in `results/optim_logs` with `.pkl` format.

#### 2. Reproduction (`reproduce.py`)
Because the optuna python library does not support saving the optimal model, we reproduce the model with optimal configuration found in optimization step. The optimal configuration is easily loaded from the optimization logs as `[optimization logs].best_params`. The reproduced performance for each dataset and algorithm-setup combination is available in `results/full_results.csv`.

### Examples
Here we provide the simple examples in step by step.
1. Clone the repository:
    ```sh
    git clone https://github.com/kyungeun.lee/multitab.git
    ```
2. Navigate to the project directory:
    ```sh
    cd multitab
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Implement hyperparameter optimization:
    ```sh
    python optimize.py --gpu_id [GPU_ID] --openml_id [DATASET_ID] --modelname [MODELNAME] --preprocessing [NUMERIC_FEATURE_PREPROCESSING] --cat_threshold [CATEGORY_THRESHOLD] --savepath [SAVEPATH]
    ```
    
    **ARGUMENTS**
    - gpu_id: GPU index for training (dtype: `int`)
    - openml_id: Dataset index (See `dataset_id.json` for detailed information). (dtype: `int`)
    - modelname: Model name (Options: xgboost, catboost, lightgbm, mlp, resnet, ftt, t2gformer) 
    - preprocessing: Numerical feature preprocessing methods (Options: standardization, quantile)
    - cat_threshold: Thresholds for defining the categorical features (dtype: `int`)
    - savepath: Path to save the results (dtype: `str`)
    
    **RESULTS**
    - Optimization logs into `.pkl` file in the predetermined `savepath`.
    
5. Reproduce the optimal configuration:
    ```sh
    python reproduce.py --gpu_id [GPU_ID] --openml_id [DATASET_ID] --modelname [MODELNAME] --preprocessing [NUMERIC_FEATURE_PREPROCESSING] --cat_threshold [CATEGORY_THRESHOLD] --ensemble [ENSEMBLE_TRIAL] --savepath [SAVEPATH]
    ```
    - gpu_id: GPU index for training (dtype: `int`)
    - openml_id: Dataset index (See `dataset_id.json` for detailed information). (dtype: `int`)
    - modelname: Model name (Options: xgboost, catboost, lightgbm, mlp, resnet, ftt, t2gformer) 
    - preprocessing: Numerical feature preprocessing methods (Options: standardization, quantile)
    - cat_threshold: Thresholds for defining the categorical features (dtype: `int`)
    - ensemble: Optional argument for allowing the repeated reproduction (dtype: `int`)
    - savepath: Path to save the results (dtype: `str`)
    
    **REQUIRED**
    - Optimization logs into `.pkl` file in the predetermined `savepath`.
    
    **RESULTS**
    - Reproduced prediction and performance into `.npy` file in the predetermined `savepath`.

### License


### Contact
Kyungeun Lee (e-mail: kyungeun.lee@lgresearch.ai)
