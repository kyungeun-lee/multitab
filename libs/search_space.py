import optuna

### Define hyperparameter search space (Supplementary E)
def get_search_space(trial, modelname):
    if modelname == "randomforest":
        params = {
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5000, 50000),
            'min_sample_leaf': trial.suggest_categorical('min_sample_leaf', [1, 2, 3, 4, 5, 10, 20, 40, 80]),
            'max_features': trial.suggest_categorical('max_features', []),
            'n_estimators': 300
        }
    elif modelname == "catboost":
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
            'early_stopping_rounds': 50,
            'od_pval': 0.001,
            'iterations': 2000,
            'verbose': 0,
#             'task_type': 'GPU',
            'one_hot_max_size': 0
        }
    elif modelname == "xgboost":
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-8, 1e5, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'gamma': trial.suggest_int('gamma', 1, 1e2, log=True),
            'lambda': trial.suggest_int('lambda', 1, 1e2, log=True),
            'alpha': trial.suggest_int('lambda', 1, 1e2, log=True),
            'early_stopping_rounds': 50,
            'od_pval': 0.001,
            'n_estimators': 100,
            'verbosity': 0
        }
    elif modelname == "lightgbm":
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 4, 64),
            'min_data_in_leaf': 40,
            'boost_from_average': True,
            'bagging_freq': 3,
            'bagging_fraction': 0.9,
            'early_stopping_rounds': 50,
            'num_iterations': 2000,
            'verbosity': -1
        }
    elif modelname == "mlp":
        params = {
            'depth': trial.suggest_int('depth', 1, 8),
            'width': trial.suggest_int('width', 1, 512),
            'dropout': trial.suggest_float('dropout', 0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3),
            'normalization': trial.suggest_categorical('normalization', [None, "batchnorm", "layernorm"]), 
            'activation': trial.suggest_categorical('activation', ["relu", "lrelu", "sigmoid", "tanh", "gelu"]), 
            'n_epochs': 100,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }        
    elif modelname == "resnet":
        params = {
            'n_layers': trial.suggest_int('n_layers', 1, 8),
            'd': trial.suggest_int('d', 64, 512),
            'd_embedding': trial.suggest_int('d_embedding', 64, 512),
            'd_hidden_factor': trial.suggest_float('d_hidden_factor', 1, 4),
            'hidden_dropout': trial.suggest_float('hidden_dropout', 0, 0.5),
            'residual_dropout': trial.suggest_float('residual_dropout', 0, 0.5),
            'activation': trial.suggest_categorical('activation', ["reglu", "geglu", "sigmoid", "relu"]),
            'normalization': trial.suggest_categorical('normalization', ["batchnorm", "layernorm"]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3),
            'n_epochs': 100,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
    elif modelname == "ftt":
        params = {
            ## tokenizer
            'token_bias': trial.suggest_categorical('token_bias', [True, False]),
            ## transformer
            'n_layers': trial.suggest_int('n_layers', 1, 4),
            'd_token': trial.suggest_int('d_token', 8, 64), ##original: trial.suggest_int('d_token', 64, 512), -> it should be divided into 8(=n_heads)
            'n_heads': 8, #do not tune in original paper
            'd_ffn_factor': trial.suggest_float('d_ffn_factor', 2/3, 8/3),
            'attention_dropout': trial.suggest_float('attention_dropout', 0, 0.5),
            'ffn_dropout': trial.suggest_float('ffn_dropout', 0, 0.5),
            'residual_dropout': trial.suggest_float('residual_dropout', 0, 0.2),
            'activation': trial.suggest_categorical('activation', ["reglu", "geglu", "sigmoid", "relu"]),
            'prenormalization': trial.suggest_categorical('prenormalization', [True, False]),
            'initialization': trial.suggest_categorical('initialization', ["xavier", "kaiming"]),
            ## linformer
            'kv_compression': None, ## default setup in original paper
            'kv_compression_sharing': None, ## default setup in original paper
            ## optimizer
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3),
            'n_epochs': 100,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
    elif modelname == "t2gformer":
        params = {
            'n_layers': trial.suggest_int('n_layers', 1, 5),
            'd_token': trial.suggest_int('d_token', 8, 64), ##original: trial.suggest_int('d_token', 64, 512), -> it should be divided into 8(=n_heads) -- same as ftt
            
            'residual_dropout': trial.suggest_float('residual_dropout', 0, 0.2),
            'attention_dropout': trial.suggest_float('attention_dropout', 0, 0.5),
            'ffn_dropout': trial.suggest_float('ffn_dropout', 0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'learning_rate_embed': trial.suggest_float('learning_rate_embed', 5e-3, 5e-2, log=True),
            'n_heads': 8, ## default setup in original paper
            'token_bias': True, ## default setup in original paper
            'kv_compression': None, ## default setup in original paper
            'kv_compression_sharing': None, ## default setup in original paper
            'd_ffn_factor': trial.suggest_float('d_ffn_factor', 2/3, 8/3),
            'prenormalization': trial.suggest_categorical('prenormalization', [True, False]),
            'initialization': trial.suggest_categorical('initialization', ["xavier", "kaiming"]),
            'activation': trial.suggest_categorical('activation', ["reglu", "geglu", "sigmoid", "relu"]),
            
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
            'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3), 
            'n_epochs': 100,
            'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]) 
        }
#     elif modelname == "saint":
#         params = {
#             'activation': trial.suggest_categorical('activation', ["reglu", "geglu", "sigmoid", "relu"]),
#             'attn_dropout',
#             'ff_dropout',
#             'cont_embeddings': trial.suggest_categorical('cont_embeddings', ['MLP','Noemb','pos_singleMLP']),
#             'attentiontype': trial.suggest_categorical('attentiontype', ['col','colrow','row','justmlp','attn','attnmlp']),
#             'final_mlp_style': trial.suggest_categorical('final_mlp_style', ['common', 'sep']),
#             'optimizer': trial.suggest_categorical('optimizer', ["AdamW", "Adam", "sgd"]),
#             'lr_scheduler': trial.suggest_categorical('lr_scheduler', [True, False]), 
#             'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3), 
#             'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
#             'n_epochs': 100
#         }
    return params