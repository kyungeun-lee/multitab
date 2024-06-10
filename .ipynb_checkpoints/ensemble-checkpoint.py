import optuna, argparse, os, torch, json, joblib, time, datetime
from libs.data import TabularDataset
from libs.tree import CatBoost, XGBoost, LightGBM
from libs.mlp import MLP
from libs.fttransformer import FTTransformer
from libs.resnet import ResNet
from libs.t2gformer import T2GFormer
from libs.eval import *
from libs.search_space import *
import pandas as pd
import warnings, shutil
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

savepath = "/home/lab-di/squads/supertab/results"

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", type=int, default=5)
parser.add_argument("--openml_id", type=int, default=4538)
parser.add_argument("--where_is_your_code", type=str, default="/home/tabsemi_v2")

parser.add_argument("--labeled_data", type=float, default=1.0)
parser.add_argument("--modelname", type=str, default='mlp', choices=['xgboost', 'catboost', 'lightgbm', 'mlp', 'ftt', 'resnet', 't2gformer'])
parser.add_argument("--trial", type=int, default=1)
parser.add_argument("--timelimit", type=int, default=1)

args = parser.parse_args()
   
if args.trial == 0:
    preprocessing = "standardization"
else:
    preprocessing = "quantile"
    
if args.trial == 2:
    cat_threshold = 0
elif args.trial == 3:
    cat_threshold = 50
elif args.modelname in ["mlp", "xgboost"]:
    cat_threshold = 0
else:
    cat_threshold = 20

with open(f'{args.where_is_your_code}/dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info.get(str(args.openml_id))['tasktype']

savepath = os.path.join(savepath, f'labeled_data={args.labeled_data}/data={args.openml_id}/model={args.modelname}/trial={args.trial}')
savepath = os.path.join(savepath, f'cat_threshold={cat_threshold}')
fname = os.path.join(savepath, 'result.pkl')
print(fname)
assert os.path.exists(fname)

if tasktype == "regression":
    ascending = True
    metric = "user_attrs_rmse_test"
else:
    ascending = False
    metric = "user_attrs_acc_test"

done_result = joblib.load(fname)
cur_result = done_result.trials_dataframe()
limit_result = cur_result[cur_result["duration"].cumsum() < pd.Timedelta(10, 'h')].sort_values("value", ascending=ascending)

if args.timelimit:
    savepath = os.path.join(savepath, 'timelimit')
else:
    savepath = os.path.join(savepath, 'ensemble-bst')
if not os.path.exists(savepath):
    os.makedirs(savepath)

repeat = 0
train = True
# if os.path.exists(os.path.join(savepath, "performance-%i.npy" %(repeat))):
#     train = False
#     exit()
if limit_result.shape[0] == 100:
    result = pd.read_csv("/home/tabsemi_v2/results/full_results_reproduced.csv", index_col=0)
    result["modelname"] = result["modelname"].replace("xgb", "xgboost")
    result["modelname"] = result["modelname"].replace("catb", "catboost")
    result["modelname"] = result["modelname"].replace("lgbm", "lightgbm")
    result["modelname"] = result["modelname"].replace("t2g", "t2gformer")
    result = result[(result["data_id"] == args.openml_id) & (result["trial"] == args.trial) & (result["modelname"] == args.modelname)]["perf"].abs().values[0]
    if tasktype == "regression":
        np.save(os.path.join(savepath, "performance-0.npy"), dict({"rmse_test": result}))
    else:
        np.save(os.path.join(savepath, "performance-0.npy"), dict({"acc_test": result}))
    train = False
    print("Nothing to do")
    exit()
    
else:
    ti = limit_result.head(1)["number"].values[0]
    
if ti == done_result.best_trial.number:
    train = False
    print("12345")
    result = pd.read_csv("/home/tabsemi_v2/results/full_results_reproduced.csv", index_col=0)
    result["modelname"] = result["modelname"].replace("xgb", "xgboost")
    result["modelname"] = result["modelname"].replace("catb", "catboost")
    result["modelname"] = result["modelname"].replace("lgbm", "lightgbm")
    result["modelname"] = result["modelname"].replace("t2g", "t2gformer")
    result = result[(result["data_id"] == args.openml_id) & (result["trial"] == args.trial) & (result["modelname"] == args.modelname)]["perf"].abs().values[0]
    
    if tasktype == "regression":
        np.save(os.path.join(savepath, "performance-0.npy"), dict({"rmse_test": result}))
    else:
        np.save(os.path.join(savepath, "performance-0.npy"), dict({"acc_test": result}))
    exit()

# if train:
#     params = done_result.trials[ti].params
#     if args.modelname in ["mlp", "resnet"]:
#         params['n_epochs'] = 100
#     elif args.modelname in ["ftt", "t2gformer"]:
#         params["n_heads"] = 8
#         params["kv_compression"] = None
#         params["kv_compression_sharing"] = None
#         params["n_epochs"] = 100
#     if args.modelname == "t2gformer":
#         params["token_bias"] = True
#     if args.modelname == "ftt":
#         params["ffn_dropout"] = params["attention_dropout"]

#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#     os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     env_info = '{0}:{1}'.format(os.uname().nodename, args.gpu_id)

#     if preprocessing == "quantile":
#         dataset = TabularDataset(args.openml_id, tasktype, device=device, labeled_data=args.labeled_data, cat_threshold=cat_threshold, modelname=args.modelname, quantile=True)
#     else:
#         dataset = TabularDataset(args.openml_id, tasktype, device=device, labeled_data=args.labeled_data, cat_threshold=cat_threshold, modelname=args.modelname)
#     (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
#     y_std = dataset.y_std.item() if tasktype == "regression" else None

#     assert torch.isnan(X_train).sum() == 0
#     assert torch.isnan(X_val).sum() == 0
#     assert torch.isnan(X_test).sum() == 0

#     if (tasktype == "multiclass") & (args.modelname in ["catboost", "xgboost", "lightgbm"]):
#         if y_train.size(1) != y_train.unique(dim=1).size(1):            
#             raise ValueError #"Unknown class problem"

#     if (args.openml_id in [
#         1567, 1492, 44129, 44159, 150, 41960, 6, 41027, 41169, 42345, 41168, 23512, 42396, 
#         44161, 41150, 1113, 41960, 150, 44065, 44027, 41147, 1503, 43928, 41275]) & (
#         args.modelname.startswith("catboost")):

#         params['task_type'] = 'GPU'
#         print("CatB with GPU")

#     if args.modelname == "catboost":
#         model = CatBoost(params, tasktype, dataset.X_cat)
#     elif args.modelname == "xgboost":
#         model = XGBoost(params, tasktype, dataset.X_cat)
#     elif args.modelname == "lightgbm":
#         model = LightGBM(params, tasktype, dataset.X_cat)
#     elif (args.modelname == "mlp") & (tasktype == "multiclass"):
#         model = MLP(params, tasktype, input_dim=X_train.size(1), output_dim=y_train.size(1), device=device, data_id=args.openml_id)
#     elif (args.modelname == "mlp"):
#         model = MLP(params, tasktype, input_dim=X_train.size(1), output_dim=1, device=device, data_id=args.openml_id)
#     elif (args.modelname == "ftt") & (tasktype == "multiclass"):
#         model = FTTransformer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=y_train.size(1), device=device, data_id=args.openml_id)
#     elif args.modelname == "ftt":
#         model = FTTransformer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=1, device=device, data_id=args.openml_id)
#     elif (args.modelname == "resnet") & (tasktype == "multiclass"):
#         model = ResNet(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=y_train.size(1), device=device, data_id=args.openml_id)
#     elif args.modelname == "resnet":
#         model = ResNet(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=1, device=device, data_id=args.openml_id)
#     elif (args.modelname == "t2gformer") & (tasktype == "multiclass"):
#         model = T2GFormer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=y_train.size(1), device=device, data_id=args.openml_id)
#     elif args.modelname == "t2gformer":
#         model = T2GFormer(params, tasktype, dataset.X_num, dataset.X_categories, input_dim=X_train.size(1), output_dim=1, device=device, data_id=args.openml_id)
#     model.fit(X_train, y_train, X_val, y_val)

#     preds_val = model.predict(X_val)
#     preds_test = model.predict(X_test)
# #     if tasktype != "regression":
# #         preds_test_prob = model.predict_proba(X_test)
# #         preds_test_logit = model.predict_proba(X_test, logit=True)

#     if tasktype == "regression":
#         val_metrics = calculate_metric(y_val*y_std, preds_val*y_std, tasktype, 'val')
#         test_metrics = calculate_metric(y_test*y_std, preds_test*y_std, tasktype, 'test')
#     else:
#         val_metrics = calculate_metric(y_val, preds_val, tasktype, 'val')
#         test_metrics = calculate_metric(y_test, preds_test, tasktype, 'test')
#     for k, v in val_metrics.items():
#         print(k, v)
#     for k, v in test_metrics.items():
#         print(k, v)

#     print(device, env_info, args.openml_id, data_info.get(str(args.openml_id))['name'], args.modelname, savepath)
#     print(val_metrics)
#     print(test_metrics)

#     print("#############################################")
#     print(env_info)
#     if args.modelname in ["xgboost", "catboost", "lightgbm"]:
#         np.save(os.path.join(savepath, "result-%i.npy" %(repeat)), preds_test)
#     else:
#         np.save(os.path.join(savepath, "result-%i.npy" %(repeat)), preds_test.detach().cpu().numpy())
#     np.save(os.path.join(savepath, "performance-%i.npy" %(repeat)), test_metrics)
# #     if tasktype != "regression":
# #         np.save(os.path.join(savepath, "logit-%i.npy" %(repeat)), preds_test_logit.detach().cpu().numpy())
#     print("#############################################")
