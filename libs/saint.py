# # Reference
# # SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
# # https://github.com/somepago/saint/blob/main/models/model.py

import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset
from libs.data import get_batch_size
import torch, logging
from tqdm import tqdm
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ff_encodings(x,B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
class CosineAnnealingLR_Warmup(object):
    def __init__(self, optimizer, warmup_epochs, T_max, iter_per_epoch, base_lr, warmup_lr, eta_min, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.warmup_iter = self.iter_per_epoch * self.warmup_epochs
        self.cosine_iter = self.iter_per_epoch * (self.T_max - self.warmup_epochs)
        self.current_iter = (self.last_epoch + 1) * self.iter_per_epoch

        self.step()

    def get_current_lr(self):
        if self.current_iter < self.warmup_iter:
            current_lr = (self.base_lr - self.warmup_lr) / self.warmup_iter * self.current_iter + self.warmup_lr
        else:
            current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * (self.current_iter-self.warmup_iter) / self.cosine_iter)) / 2
        return current_lr

    def step(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        self.current_iter += 1
        

def CosineAnnealingParam(warmup_epochs, T_max, iter_per_epoch, current_iter, base_value, 
                         warmup_value=1e-8, eta_min=0):
    warmup_iter = iter_per_epoch * warmup_epochs
    cosine_iter = iter_per_epoch * (T_max - warmup_epochs)
    
    if current_iter < warmup_iter:
        return (base_value - warmup_value) / warmup_iter * current_iter + warmup_value
    else:
        return eta_min + (base_value - eta_min) * (1 + np.cos(np.pi * (current_iter - warmup_iter) / cosine_iter)) / 2
# classes
class CustomTensorDataset(Dataset):
    def __init__(self, X, y, cat_cols, con_cols, task='clf'):
        super().__init__()
       
        self.y = y
        self.task = task
        self.cls = torch.zeros((self.y.shape[0], 1), dtype=torch.int64, device='cuda')
        self.X1 = X[:, cat_cols].to(dtype=torch.int64)
        self.X2 = X[:, con_cols].to(dtype=torch.float32)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.cls[idx].dim() == 1:  # 슬라이스 또는 여러 인덱스
            return torch.cat((self.cls[idx], self.X1[idx]), dim=0), self.X2[idx], self.y[idx]
        else:  # 단일 인덱스
            return torch.cat((self.cls[idx], self.X1[idx]), dim=1), self.X2[idx], self.y[idx]
        
       

        # return torch.cat((self.cls[idx], self.X1[idx]), dim=0), self.X2[idx], self.y[idx]
    def get_batches(self, batch_size):
        """Return data in batches in the same format as __getitem__"""
        n_samples = len(self.y)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            # Get the batches for each component
            cls_batch = self.cls[i:end_idx]
            X1_batch = self.X1[i:end_idx]
            X2_batch = self.X2[i:end_idx]
            y_batch = self.y[i:end_idx]
            
            # Combine cls and X1 just like in __getitem__
            cat_with_cls = torch.cat((cls_batch, X1_batch), dim=0)
            
            yield cat_with_cls, X2_batch, y_batch

def embed_data(x_categ, x_cont, model):
    device = x_cont.device
    # import IPython; IPython.embed()
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1, n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')    

    x_categ_enc = x_categ_enc.to(device)
    x_cont_enc = x_cont_enc.to(device)
    
    return x_categ, x_categ_enc, x_cont_enc

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, x_num,
                x_cat, heads, dim_head, attn_dropout, ff_dropout,style='col'):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed =  nn.Embedding(nfeats, dim)
        self.style = style
        self.x_num = x_num
        self.x_cat = x_cat
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))

    def forward(self, x_categ_enc, x_cont_enc, mask = None):
        
        if x_categ_enc is not None:
            x = torch.cat((x_categ_enc,x_cont_enc),dim=1)
        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers: 
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        else:
             for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)    

#mlp
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(GEGLU())
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class simple_MLP(nn.Module):
    def __init__(self,dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# main class

class TabAttention(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 1,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        lastmlp_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'colrow'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories_index)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.x_cat = categories
        
        self.x_num = num_continuous
        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories_index)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                x_num = self.x_num,
                x_cat = self.x_cat,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

    def forward(self, x_categ, x_cont,x_categ_enc,x_cont_enc):
        device = x_categ.device
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim = -1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == 'MLP':
                
                x = self.transformer(x_categ_enc.to(device),x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else: 
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim = -1)                    
        flat_x = x.flatten(1)
        return self.mlp(flat_x)


class sep_MLP(nn.Module):
    def __init__(self,dim,len_feats,categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))

        
    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred

class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        categories_index,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 0,
        attn_dropout = 0.,
        ff_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col',
        final_mlp_style = 'common',
        y_dim = 2
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories_index)), 'number of each category must be positive'
        # categories related calculations
        # import IPython; IPython.embed()
        self.num_categories = len(categories_index)
        self.num_unique_categories = sum(categories_index)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        
        categories_offset = F.pad(torch.tensor(list(categories_index)), (1, 0), value = num_special_tokens)
        self.categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        if not hasattr(self, 'categories_offset'):
            self.register_buffer('categories_offset', self.categories_offset)

        self.x_cat = categories
       
        self.x_num = num_continuous
        
        self.num_continuous = len(num_continuous)
        self.norm = nn.LayerNorm(self.num_continuous)
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

     
        self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
        input_size = (dim * self.num_categories)  + (dim * self.num_continuous)
        nfeats = self.num_categories + self.num_continuous


        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                x_num = self.x_num,
                x_cat = self.x_cat,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)


        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)
        
        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim,(self.total_tokens)*2, self.total_tokens])
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim,self.num_categories,categories_index)
            self.mlp2 = sep_MLP(dim,self.num_continuous,np.ones(self.num_continuous).astype(int))


        self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.pt_mlp = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])

        
    def forward(self, x1, x2):

        x = self.transformer(x1, x2)

        x_cls = x[:,0,:]
        x_out = self.mlpfory(x_cls)
        return x_out 
    
### added
from libs.supervised import supmodel
class build_saint(SAINT):
    def __init__(self, 
                 categories, num_continuous, categories_index, dim, depth, heads, dim_head=16, dim_out=1, mlp_hidden_mults=(4, 2), mlp_act=None,
                 num_special_tokens=0, attn_dropout=0., ff_dropout=0., cont_embeddings='MLP', scalingfactor=10,
                 attentiontype='colrow', final_mlp_style='common', y_dim=2, optimizer="AdamW", learning_rate=0., weight_decay=0.):
        
        # Initialize the parent class (SAINT) with its parameters
        super().__init__(categories=categories, num_continuous=num_continuous, categories_index=categories_index, dim=dim, depth=depth, heads=heads, 
                         dim_head=dim_head, dim_out=dim_out, mlp_hidden_mults=mlp_hidden_mults, mlp_act=mlp_act,
                         num_special_tokens=num_special_tokens, attn_dropout=attn_dropout, ff_dropout=ff_dropout, 
                         cont_embeddings=cont_embeddings, scalingfactor=scalingfactor,
                         attentiontype=attentiontype, final_mlp_style=final_mlp_style, y_dim=y_dim)

        # Additional attributes for build_saint related to optimization
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Instantiate the optimizer
        self.optimizer = self.make_optimizer()

    def forward(self, x1, x2):
        """
        Forward pass that leverages the SAINT model's forward method.
        """
        
        return super().forward(x1, x2)
    
    def make_optimizer(self):
        """
        Method to create an optimizer based on the optimizer_name attribute.
        """
        if self.optimizer_name == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def configure_optimizers(self):
        """
        Returns the optimizer. This method can be extended for complex optimization setups.
        """
        return self.optimizer
#categories_index = cardinarlity of categorical features
class main_saint(supmodel):
    def __init__(self, params, tasktype, num_cols=[], categories=[], categories_index=[], input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="saint"):
        
        super().__init__(tasktype, params, device, data_id, modelname, categories)

        # import IPython; IPython.embed()
        self.num_cols = num_cols
        self.categories = categories
        categories_index = np.append(np.array([1]),np.array(categories_index)).astype(int)
        embedding_dim = params.get("embedding_dim", 32)
        attn_dropout = params["attn_dropout"] if "attn_dropout" in params else params["attention_dropout"]
        ff_dropout = params["ff_dropout"] if "ff_dropout" in params else params["attention_dropout"]
        self.model = build_saint(categories, num_cols, categories_index, embedding_dim, params["depth"], params["heads"], params["hidden"], 
                                 mlp_hidden_mults=(4, 2), # fixed in official codes
                                 mlp_act=params["activation"], attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                 cont_embeddings=params.get("cont_embeddings", "MLP"), attentiontype=params.get("attentiontype", "colrow"), final_mlp_style=params["final_mlp_style"], y_dim=output_dim,
                                 optimizer=params["optimizer"], learning_rate=params["learning_rate"], weight_decay=params["weight_decay"]) 
        
        self.model = self.model.to(device)
    
    def fit(self, X_train, y_train, X_val, y_val):
        
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0])]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
            X_val = X_val[~torch.isnan(y_val[:, 0])]
            y_val = y_val[~torch.isnan(y_val[:, 0])]

        else:
            X_train = X_train[~torch.isnan(y_train)]
            y_train = y_train[~torch.isnan(y_train)]
            X_val = X_val[~torch.isnan(y_val)]
            y_val = y_val[~torch.isnan(y_val)]
            
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_val = y_val.unsqueeze(1)
        
        batch_size = get_batch_size(len(X_train))
            
        optimizer = self.model.make_optimizer()
        if self.tasktype == "regression":
            loss_fn = torch.nn.functional.mse_loss
        elif self.tasktype == "binclass":
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            loss_fn = torch.nn.functional.cross_entropy
            
        train_dataset = CustomTensorDataset(X_train, y_train, cat_cols=self.categories, con_cols=self.num_cols)
        val_dataset = CustomTensorDataset(X_val, y_val, cat_cols=self.categories, con_cols=self.num_cols)
        del X_train, y_train, X_val, y_val
        
        if len(train_dataset) % batch_size == 1:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) ## prevent error for batchnorm
        else:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer.zero_grad(); optimizer.step()
        
        if self.params["lr_scheduler"]:
            scheduler = CosineAnnealingLR_Warmup(optimizer, base_lr=self.params['learning_rate'], warmup_epochs=10, 
                                                 T_max=self.params.get('n_epochs'), iter_per_epoch=len(train_loader), 
                                                 warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        
        self.model = self.model.to(self.device)
        
        loss_history = []
        pbar = tqdm(range(1, self.params.get('n_epochs', 0) + 1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)

            import time
            st = time.time()
            for i, (x1, x2, y) in enumerate(train_loader):
                self.model.train(); optimizer.zero_grad()

                _, x_categ_enc, x_cont_enc = embed_data(x1, x2, self.model)
                out = self.model(x_categ_enc, x_cont_enc)
                if out.size() != y.size():
                    out = out.view(y.size())
                loss = loss_fn(out, y.to(self.device))
                loss_history.append(loss.item())
                
                loss.backward()
                optimizer.step() 
                if self.params["lr_scheduler"]:
                    scheduler.step()
                
                pbar.set_postfix_str(f'data_id: {self.data_id}, Model: {self.modelname}, Tr loss: {loss:.5f}')
            
            # validation loop
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x1_val, x2_val, y_val in val_loader:
                    _, x_categ_enc, x_cont_enc = embed_data(x1_val, x2_val, self.model)
                    out = self.model(x_categ_enc.to(self.device), x_cont_enc.to(self.device))
                    if out.size() != y_val.size():
                        out = out.view(y_val.size())
                    val_loss += loss_fn(out, y_val.to(self.device)).item()
            val_loss /= len(val_loader)
            
            self._callback_container.on_epoch_end(epoch, {"val_loss": val_loss, "epoch": epoch})
            if any([cb.should_stop for cb in self._callback_container.callbacks]):
                print(f"Early stopping at epoch {epoch}")
                break
                
        self.model.eval()
            
    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            # Create dataset with proper feature separation
            test_dataset = CustomTensorDataset(
                X=X_test, 
                y=np.zeros(len(X_test)),  # Dummy labels for test data
                cat_cols=self.categories,
                con_cols=self.num_cols,
                task=self.tasktype
            )
            
            if (X_test.shape[0] > 10000) or (X_test.shape[1] > 240):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    start_idx = i * 100
                    end_idx = min((i + 1) * 100, X_test.shape[0])
                    cat_features = test_dataset[start_idx:end_idx][0].to(self.device)
                    cont_features = test_dataset[start_idx:end_idx][1].to(self.device)
                    _, x_categ_enc, x_cont_enc = embed_data(cat_features, cont_features, self.model)
                    pred = self.model(x_categ_enc, x_cont_enc)
                    logits.append(pred.cpu())
                    del pred
                logits = torch.cat(logits, dim=0)
            else:
                
                cat_features, cont_features, _ = test_dataset[:]  # Get all data
                cat_features = cat_features.to(self.device)
                cont_features = cont_features.to(self.device)
                _, x_categ_enc, x_cont_enc = embed_data(cat_features, cont_features, self.model)
                logits = self.model(x_categ_enc, x_cont_enc).cpu()
                
            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().numpy()
            elif self.tasktype == "regression":
                return logits.numpy()
            else:
                return torch.argmax(logits, dim=1).numpy()    
    def predict_proba(self, X_test, logit=False):

        self.model.eval()
        with torch.no_grad():
            test_dataset = CustomTensorDataset(
                X=X_test, 
                y=np.zeros(len(X_test)),  # Dummy labels for test data
                cat_cols=self.categories,
                con_cols=self.num_cols,
                task=self.tasktype
            )
            
            if (X_test.shape[0] > 10000) or (X_test.shape[1] > 240):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    start_idx = i * 100
                    end_idx = min((i + 1) * 100, X_test.shape[0])
                    cat_features = test_dataset[start_idx:end_idx][0].to(self.device)
                    cont_features = test_dataset[start_idx:end_idx][1].to(self.device)
                    _, x_categ_enc, x_cont_enc = embed_data(cat_features, cont_features, self.model)
                    pred = self.model(x_categ_enc, x_cont_enc)
                    logits.append(pred.cpu())
                    del pred
                logits = torch.cat(logits, dim=0)
            else:
                cat_features, cont_features, _ = test_dataset[:]  # Get all data
                cat_features = cat_features.to(self.device)
                cont_features = cont_features.to(self.device)
                _, x_categ_enc, x_cont_enc = embed_data(cat_features, cont_features, self.model)
                logits = self.model(x_categ_enc, x_cont_enc).cpu()
                    
            if logit:
                return logits.numpy()
            else:
                return torch.nn.functional.softmax(logits, dim=1).numpy()