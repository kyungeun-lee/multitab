# MultiTab
The implementation codes of the paper "MultiTab: A Comprehensive Benchmark Suite for Multi-Dimensional Evaluation in Tabular Domains"
(Paper URL: TBU)

## Table of Contents
- [Overview](#overview)
- [Required libraries](#required-libraries)
- [Dataset descriptions](#dataset-descriptions)
- [Experiments](#experiments)
- [Examples](#examples)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Overview
`MultiTab` is a benchmark suite and evaluation framework designed to enable structured, data-aware analysis of tabular learning algorithms. It includes 196 datasets spanning both classification and regression tasks. `MultiTab` categorizes the entire data collection into diverse sub-categories based on seven specific characteristics of tabular data, such as data size, function irregularity, and feature interaction. For prediction algorithms, `MultiTab` incorporates 13 algorithms. Comprehensive hyperparameter tuning is performed for each algorithm and dataset to determine the optimal configurations. All optimization logs and results are available on `LGAI-DILab/Multitab`.

See our paper at [TBU].

## Required libraries
Our project has been built on Python 3.9. Here is the entire list of python libraries required (also available in `requirements.txt`):

``` swift
optuna==3.5.0
argparse==1.1
torch==2.0.1
joblib==1.2.0
scikit-learn==1.2.2
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
Our benchmark suite encompasses a comprehensive collection of 196 datasets, all of which are publicly accessible through the OpenML or scikit-learn python library. All datasets are provided under the CC-BY license, which implies that the data is publicly available and has been shared with the appropriate consent and ethical considerations. OpenML ensures that datasets shared on their platform comply with their data-sharing guidelines, which include obtaining necessary consent where applicable.

Each dataset can be loaded by inserting the dataset IDs in 
``` swift
openml.datasets.get_dataset(DATASET_ID)
```
Due to its widespread use in numerous studies, we allow an exception for the CA dataset (ID: 999999) in the regression task, which we load from the sklearn python library instead of OpenML repository. Each dataset has exactly one train-validation-test split (8:1:1 with a fixed random seed), so all algorithms use the same splits of datasets.

Here is the complete list of 210 datasets used in this study:

> 25 461 210 466 42665 444 497 10 1099 48 338 40916 23381 4153 505 560 51 566 452 53 524 49 194 42370 511 509 456 8 337 59 35 42360 455 475 40496 531 1063 703 534 1467 44968 42 1510 334 549 11 188 29 470 43611 40981 45102 1464 1549 37 43962 469 458 54 45545 50 307 1555 31 1494 4544 41702 934 1479 41021 41265 185 454 1462 43466 23 43919 42931 1501 1493 1492 1504 315 20 12 14 16 22 18 1067 1466 36 1487 44091 42727 43926 41143 507 46 3 44055 44061 1043 44160 44158 40900 44124 1489 1497 40499 24 41145 1475 182 44136 43986 503 372 44157 558 44132 562 189 40536 44056 422 44054 4538 45062 44145 44122 1531 1459 44126 44062 42183 32 4534 42734 44125 44123 44137 1476 44005 1471 44133 846 44134 44162 44089 44063 44026 45012 6 44090 537 999999 44148 44066 44984 4135 1486 45714 44064 344 41027 151 44963 40985 45068 44059 44131 40685 45548 41169 41162 42345 41168 40922 23512 40672 44161 41150 1509 44057 43928 44069 1503 44068 44159 1113 1169 150 44065 44129 1567

A detailed description of the criteria of the `MultiTab` is available in Appendix B-D of the paper.

### Experiments
We provide the implementation codes for (1) optimizing the best hyperparameter optimization using the Tree-structured Parzen Estimator provided by the optuna python library for 100 trials without exception, (2) reproducing the best configuration found in (1), (3) and ensemble the predictions.

#### 1. Optimization (`optimize.py`)
To ensure that each prediction algorithm reaches its optimal configuration, we conducted an extensive hyperparameter optimization across all datasets and algorithms, performing 100 trials without exception. The optimization objectives were tailored to the task types: validation accuracy for classification tasks and validation RMSE (Root Mean Squared Error) for regression tasks. The hyperparameter search space is described in Supplementary E of the paper and `libs/search_space.py`. As a result, the optimization logs will be saved in the predetermined directory. We provide the entire optimization logs in `results/optim_logs` with `.pkl` format.

#### 2. Reproduction (`reproduce.py`)
Because the optuna python library does not support saving the optimal model, we reproduce the model with the optimal configuration found in the optimization step. The optimal configuration is easily loaded from the optimization logs as `[optimization logs].best_params`. The reproduced performance for each dataset and the algorithm-setup combination is available in `results/full_results.csv`.

#### 3. Ensemble (`ensemble.py`)

### Examples
Here we provide simple examples step by step.
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
    python optimize.py --gpu_id [GPU_ID] --openml_id [DATASET_ID] --modelname [MODELNAME] --savepath [SAVEPATH]
    ```
    
    **ARGUMENTS**
    - gpu_id: GPU index for training (dtype: `int`)
    - openml_id: Dataset index (See `dataset_id.json` for detailed information). (dtype: `int`)
    - modelname: Model name (Options: xgboost, catboost, lightgbm, mlp, resnet, ftt, t2gformer) 
    - savepath: Path to save the results (dtype: `str`)
    
    **RESULTS**
    - Optimization logs into `.pkl` file in the predetermined `savepath`.
    
5. Reproduce the optimal configuration:
    ```sh
    python reproduce.py --gpu_id [GPU_ID] --openml_id [DATASET_ID] --seed [ENSEMBLE_TRIAL] --savepath [SAVEPATH]
    ```
    - gpu_id: GPU index for training (dtype: `int`)
    - openml_id: Dataset index (See `dataset_id.json` for detailed information). (dtype: `int`)
    - seed: Optional argument for allowing the repeated reproduction (dtype: `int`)
    - savepath: Path to save the results (dtype: `str`)
    
    **REQUIRED**
    - Optimization logs into `.pkl` file in the predetermined `savepath`.
    
    **RESULTS**
    - Reproduced prediction and performance into `.npy` file in the predetermined `savepath`.

6(Optional). Ensemble the predictions:
    ```sh
    python ensemble.py
    ```

### License
Apache-2.0 license

### Contact
Kyungeun Lee (e-mail: kyungeun.lee@lgresearch.ai)

### References
- Y Gorishniy et al., Revisiting Deep Learning Models for Tabular Data, NeurIPS, 2021 (https://github.com/yandex-research/rtdl)
- J Yan et al., T2G-FORMER: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction, AAAI, 2023 (https://github.com/jyansir/t2g-former?tab=readme-ov-file)
- HJ Ye et al., Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later, ICLR, 2025
- TALENT library (https://github.com/LAMDA-Tabular/TALENT)
