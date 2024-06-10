from libs.tree import CatBoost, XGBoost, LightGBM
from libs.mlp import MLP
from libs.fttransformer import FTTransformer
from libs.resnet import ResNet
from libs.t2gformer import T2GFormer

def getmodel(modelname, params, tasktype, dataset, openml_id):
    
    if  == "catboost":
        model = CatBoost(params, tasktype, dataset.X_cat)
    elif  == "xgboost":
        model = XGBoost(params, tasktype, dataset.X_cat)
    elif  == "lightgbm":
        model = LightGBM(params, tasktype, dataset.X_cat)
    elif ( == "mlp") & (tasktype == "multiclass"):
        model = MLP(params, tasktype, input_dim=X_train.size(1), output_dim=y_train.size(1), device=device, data_id=openml_id)
    elif ( == "mlp"):
        model = MLP(params, tasktype, input_dim=X_train.size(1), output_dim=1, device=device, data_id=openml_id)
    elif ( == "ftt") & (tasktype == "multiclass"):
        model = FTTransformer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=y_train.size(1), device=device, data_id=openml_id)
    elif  == "ftt":
        model = FTTransformer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=1, device=device, data_id=openml_id)
    elif ( == "resnet") & (tasktype == "multiclass"):
        model = ResNet(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=y_train.size(1), device=device, data_id=openml_id)
    elif  == "resnet":
        model = ResNet(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=1, device=device, data_id=openml_id)
    elif ( == "t2gformer") & (tasktype == "multiclass"):
        model = T2GFormer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=y_train.size(1), device=device, data_id=openml_id)
    elif  == "t2gformer":
        model = T2GFormer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=1, device=device, data_id=openml_id)
        
    return model