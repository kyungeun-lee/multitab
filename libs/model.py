from libs.tree import LR, KNN, DecisionTree, RandomForest, CatBoost, XGBoost, LightGBM
from libs.mlp import MLP
from libs.embedmlp import embedMLP
from libs.mlpplr import embedMLPPLR
from libs.fttransformer import FTTransformer
from libs.resnet import ResNet
from libs.t2gformer import T2GFormer
from libs.tabpfn import tabpfn
from libs.saint import main_saint
from libs.modernnca import ModernNCAMethod
from libs.tabr import TabRMethod
from libs.search_space import get_search_space

def getmodel(modelname, params, tasktype, dataset, openml_id, input_dim, output_dim, device):
    
    if modelname == "lr":
        model = LR(tasktype)
    elif modelname == "knn":
        model = KNN(tasktype)
    elif modelname == "dt":
        model = DecisionTree(tasktype)
    elif modelname == "randomforest":
        model = RandomForest(params, tasktype)
    elif modelname == "catboost":
        model = CatBoost(params, tasktype, dataset.X_cat)
    elif modelname == "xgboost":
        model = XGBoost(params, tasktype, dataset.X_cat)
    elif modelname == "lightgbm":
        model = LightGBM(params, tasktype, dataset.X_cat)
    elif (modelname == "mlp") & (tasktype == "multiclass"):
        model = MLP(params, tasktype, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif (modelname == "mlp"):
        model = MLP(params, tasktype, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif (modelname == "embedmlp") & (tasktype == "multiclass"):
        model = embedMLP(params, tasktype, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id, cat_cols=dataset.X_cat, num_cols=dataset.X_num, categories=dataset.X_cat_cardinality)
    elif (modelname == "embedmlp"):
        model = embedMLP(params, tasktype, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id, cat_cols=dataset.X_cat, num_cols=dataset.X_num, categories=dataset.X_cat_cardinality)
    elif (modelname == "mlpplr") & (tasktype == "multiclass"):
        model = embedMLPPLR(params, tasktype, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id, cat_cols=dataset.X_cat, num_cols=dataset.X_num, categories=dataset.X_cat_cardinality)
    elif (modelname == "mlpplr"):
        model = embedMLPPLR(params, tasktype, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id, cat_cols=dataset.X_cat, num_cols=dataset.X_num, categories=dataset.X_cat_cardinality)
    elif (modelname == "ftt") & (tasktype == "multiclass"):
        model = FTTransformer(params, tasktype, dataset.X_num, dataset.X_cat_cardinality, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "ftt":
        model = FTTransformer(params, tasktype, dataset.X_num, dataset.X_cat_cardinality, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif (modelname == "resnet") & (tasktype == "multiclass"):
        model = ResNet(params, tasktype, dataset.X_num, dataset.X_cat_cardinality, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "resnet":
        model = ResNet(params, tasktype, dataset.X_num, dataset.X_cat_cardinality, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif modelname == "tabpfn":
        model = tabpfn(tasktype)
    elif (modelname == "t2gformer") & (tasktype == "multiclass"):
        model = T2GFormer(params, tasktype, dataset.X_num, dataset.X_cat_cardinality, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "t2gformer":
        model = T2GFormer(params, tasktype, dataset.X_num, dataset.X_cat_cardinality, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif (modelname == "saint") & (tasktype == "multiclass"):
        model = main_saint(params, tasktype, dataset.X_num, dataset.X_cat, dataset.X_cat_cardinality, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "saint":
        model = main_saint(params, tasktype, dataset.X_num, dataset.X_cat, dataset.X_cat_cardinality, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif (modelname == "modernnca") & (tasktype == "multiclass"):
        model = ModernNCAMethod(params, tasktype, dataset.X_num, dataset.X_cat, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "modernnca":
        model = ModernNCAMethod(params, tasktype, dataset.X_num, dataset.X_cat, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif (modelname == "tabr")  & (tasktype == "multiclass"):
        model = TabRMethod(params, tasktype, dataset.X_num, dataset.X_cat, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "tabr":
        model = TabRMethod(params, tasktype, dataset.X_num, dataset.X_cat, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    return model

def add_default_params(modelname, params, data_id=None):
    if modelname == "randomforest":
        params["n_estimators"] = 300
    elif modelname == "xgboost":
        params["early_stopping_rounds"] = 20
        params["n_estimators"] = 10000
        params["max_iterations"] = 10000
        params["verbosity"] = 0
    elif modelname == "catboost":
        params["early_stopping_rounds"] = 20
        params["iterations"] = 10000
        params["verbose"] = 0
    elif modelname == "lightgbm":
        params["early_stopping_rounds"] = 20
        params["iterations"] = 10000
        params["verbosity"] = -1
    elif modelname in ["mlp", "embedmlp", "mlpplr", "resnet"]:
        params["n_epochs"] = 100
        params["early_stopping_rounds"] = 20
    elif modelname == "ftt":
        params["n_heads"] = 8
        params["kv_compression"] = None
        params["kv_compression_sharing"] = None
        params["n_epochs"] = 100
        params["early_stopping_rounds"] = 20
    elif modelname == "t2gformer":
        params["n_heads"] = 8
        params["token_bias"] = True
        params["kv_compression"] = None
        params["kv_compression_sharing"] = None
        params["n_epochs"] = 100
        params["early_stopping_rounds"] = 20
    elif modelname == "saint":
        large_datalist = [44159, 1113, 44027, 41960, 1169, 150, 44065, 44129, 1567, 5, 20, 12, 41147, 422]
        params["depth"] = 3 if data_id in large_datalist else 6
        params["heads"] = 4 if data_id in large_datalist else 8
        params["hidden"] = 16
        params["attentiontype"] = "colrow"
        params["n_epochs"] = 100
        params["early_stopping_rounds"] = 20
        if data_id in [5, 1486, 1501, 20, 12, 41143, 44061, 1476, 41702, 41145, 41147, 422]:
            params["embedding_dim"] = 8
    return params