# # Reference
# # Revisiting deep learning models for tabular data (Y Gorishniy et al., NeurIPS 2021)
# # https://github.com/yandex-research/rtdl

from libs.supervised import supmodel
import torch, math
from rtdl_num_embeddings import PeriodicEmbeddings

class embedmlpplr(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 depth, width, dropout, normalization, activation,
                 optimizer, learning_rate, weight_decay, cat_cols, num_cols, categories, d_embedding_cat, d_embedding_num):
        
        super(embedmlpplr, self).__init__()
        
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.categories = categories
        d_numerical = len(num_cols)
        input_dim = d_numerical + len(categories)

        self.cont_embeddings = PeriodicEmbeddings(len(num_cols), d_embedding_num, lite=False)
        
        if normalization == "batchnorm":
            normalize_fn = torch.nn.BatchNorm1d(width)
        elif normalization == "layernorm":
            normalize_fn = torch.nn.LayerNorm(width)
        elif normalization == None:
            normalize_fn = torch.nn.Identity()
        
        if activation == "relu":
            act_fn = torch.nn.ReLU()
        elif activation == "lrelu":
            act_fn = torch.nn.LeakyReLU(negative_slope=0.01)
        elif activation == "sigmoid":
            act_fn = torch.nn.Sigmoid()
        elif activation == "tanh":
            act_fn = torch.nn.Tanh()
        elif activation == "gelu":
            act_fn = torch.nn.GELU()
    
        d_in = d_numerical * d_embedding_num
        if len(categories) > 0:
            d_in += len(categories) * d_embedding_cat
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = torch.nn.Embedding(sum(categories), d_embedding_cat)
            torch.nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        
        model = [torch.nn.Linear(d_in, width), act_fn]
        for _ in range(depth-1):
            model.append(torch.nn.Linear(width, width))
            model.append(normalize_fn)
            model.append(act_fn)
            model.append(torch.nn.Dropout(dropout))
        model.append(torch.nn.Linear(width, output_dim))
        
        self.model = torch.nn.Sequential(*model)
        
    def forward(self, x):
        
        x_num = x[:, self.num_cols] if len(self.num_cols) > 0 else None
        x_cat = x[:, self.cat_cols] if len(self.cat_cols) > 0 else None
        
        x = []
        if x_num is not None:
            x.append(self.cont_embeddings(x_num).reshape(x_num.size(0), -1))
        if x_cat != None:
            emb = x_cat + self.category_offsets[None]
            x.append(self.category_embeddings(emb.to(torch.long)).view(x_cat.size(0), -1))
        x = torch.cat(x, dim=-1)
        
        return self.model(x)
        
    def make_optimizer(self):
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        
class build_embedmlpplr(embedmlpplr):
    def __init__(self, 
                 input_dim, output_dim, depth, width, dropout, normalization, activation,
                 optimizer, learning_rate, weight_decay, cat_cols, num_cols, categories, d_embedding_cat, d_embedding_num):
        
        super().__init__(input_dim, output_dim, depth, width, dropout, normalization, activation, optimizer, learning_rate, weight_decay, cat_cols, num_cols, categories, d_embedding_cat, d_embedding_num)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = embedmlpplr(input_dim, output_dim, depth, width, dropout, normalization, activation, optimizer, learning_rate, weight_decay, cat_cols, num_cols, categories, d_embedding_cat, d_embedding_num)
    
    def forward(self, x, cat_features=[]):
        return self.model(x)
    
    def make_optimizer(self):
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)

    
class embedMLPPLR(supmodel):
    def __init__(self, params, tasktype, num_cols=[], cat_cols=[], input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="mlpplr", categories=[]):
        
        super().__init__(tasktype, params, device, data_id, modelname)
        self.model = build_embedmlpplr(
            input_dim, output_dim, params["depth"], params["width"], 
            params["dropout"], params["normalization"], params["activation"], params["optimizer"], params["learning_rate"], 
            params["weight_decay"], cat_cols, num_cols, categories, params["d_embedding_cat"], params["d_embedding_num"])
        self.model = self.model.to(device)
    
    def fit(self, X_train, y_train, X_val, y_val):
                    
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_val = y_val.unsqueeze(1)
        
        super().fit(X_train, y_train, X_val, y_val)
            
    def predict(self, X_test):
        return super().predict(X_test)
        
    def predict_proba(self, X_test, logit=False):
        return super().predict_proba(X_test, logit=logit)