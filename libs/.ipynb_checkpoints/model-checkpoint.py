from libs.tree import CatBoost, XGBoost, LightGBM
from libs.mlp import MLP
from libs.fttransformer import FTTransformer
from libs.resnet import ResNet
from libs.t2gformer import T2GFormer

def getmodel(modelname, params, tasktype, dataset, openml_id, input_dim, output_dim):
    
    if modelname == "catboost":
        model = CatBoost(params, tasktype, dataset.X_cat)
    elif modelname == "xgboost":
        model = XGBoost(params, tasktype, dataset.X_cat)
    elif modelname == "lightgbm":
        model = LightGBM(params, tasktype, dataset.X_cat)
    elif (modelname == "mlp") & (tasktype == "multiclass"):
        model = MLP(params, tasktype, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif (modelname == "mlp"):
        model = MLP(params, tasktype, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif (modelname == "ftt") & (tasktype == "multiclass"):
        model = FTTransformer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "ftt":
        model = FTTransformer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif (modelname == "resnet") & (tasktype == "multiclass"):
        model = ResNet(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "resnet":
        model = ResNet(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
    elif (modelname == "t2gformer") & (tasktype == "multiclass"):
        model = T2GFormer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=input_dim, output_dim=output_dim, device=device, data_id=openml_id)
    elif modelname == "t2gformer":
        model = T2GFormer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=input_dim, output_dim=1, device=device, data_id=openml_id)
        
    return model


def add_default_params(modelname, params):
    if modelname in ["mlp", "resnet"]:
        params['n_epochs'] = 100
    elif modelname in ["ftt", "t2gformer"]:
        params["n_heads"] = 8
        params["kv_compression"] = None
        params["kv_compression_sharing"] = None
        params["n_epochs"] = 100
    if modelname == "t2gformer":
        params["token_bias"] = True
    if modelname == "ftt":
        params["ffn_dropout"] = params["attention_dropout"]
    return params