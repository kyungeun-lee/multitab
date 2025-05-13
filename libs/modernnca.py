import torch
import torch.nn as nn
import torch.nn.functional as F

import abc
import math, time
import statistics
from functools import partial
from typing import Any, Callable, Optional, Union, cast
from torch import Tensor
from torch.nn.parameter import Parameter
from tqdm import tqdm

from libs.data import get_batch_size
from libs.supervised import CallbackContainer, EarlyStopping, CosineAnnealingLR_Warmup

class Averager():
    """
    A simple averager.

    """
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        """
        
        :x: float, value to be added
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def _initialize_embeddings(weight: Tensor, d: Optional[int]) -> None:
    if d is None:
        d = weight.shape[-1]
    d_sqrt_inv = 1 / math.sqrt(d)
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)

def make_trainable_vector(d: int) -> Parameter:
    x = torch.empty(d)
    _initialize_embeddings(x, None)
    return Parameter(x)

class CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = make_trainable_vector(d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        assert x.shape[-1] == len(self.weight)
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)

class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x

class PLREmbeddings(nn.Sequential):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.

    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                # if lite
                # else NLinear(n_features, 2 * n_frequencies, d_embedding)
            ),
            nn.ReLU(),
        )
        
_CUSTOM_MODULES = {
    x.__name__: x
    for x in [
        PLREmbeddings
    ]
}

class Residual_block(nn.Module):
    def __init__(self,d_in,d,dropout):
        super().__init__()
        self.linear0=nn.Linear(d_in,d)
        self.Linear1=nn.Linear(d,d_in)
        self.bn=nn.BatchNorm1d(d_in)
        self.dropout=nn.Dropout(dropout)
        self.activation=nn.ReLU()
    def forward(self, x):
        z=self.bn(x)
        z=self.linear0(z)
        z=self.activation(z)
        z=self.dropout(z)
        z=self.Linear1(z)
        # z=x+z
        return z

class ModernNCA(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_num:int,
        d_out: int,
        dim:int,
        dropout:int,
        d_block:int,
        n_blocks:int,
        num_embeddings: Optional[dict],
        temperature:float=1.0,
        sample_rate:float=0.8
        ) -> None:

        super().__init__()
        self.d_in = d_in if num_embeddings is None else d_num*num_embeddings['d_embedding']+d_in-d_num      
        self.d_out = d_out
        self.d_num=d_num
        self.dim = dim
        self.dropout = dropout
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.T=temperature
        self.sample_rate=sample_rate
        if n_blocks >0:
            self.post_encoder=nn.Sequential()
            for i in range(n_blocks):
                name=f"ResidualBlock{i}"
                self.post_encoder.add_module(name,self.make_layer())
            self.post_encoder.add_module('bn',nn.BatchNorm1d(dim))
        self.encoder = nn.Linear(self.d_in, dim)
        
        self.num_embeddings = (
            None
            if num_embeddings is None
            else PLREmbeddings(**num_embeddings, n_features=d_num)
        )

    def make_layer(self):
        block=Residual_block(self.dim,self.d_block,self.dropout)
        return block

    def forward(self, x, y,
                candidate_x, candidate_y, is_train,
                ):
        if is_train:
            data_size=candidate_x.shape[0]
            retrival_size=int(data_size*self.sample_rate)
            sample_idx=torch.randperm(data_size)[:retrival_size]
            candidate_x=candidate_x[sample_idx]
            candidate_y =candidate_y[sample_idx]
        if self.num_embeddings is not None and self.d_num >0:
            x_num,x_cat=x[:,:self.d_num],x[:,self.d_num:]
            candidate_x_num,candidate_x_cat=candidate_x[:,:self.d_num],candidate_x[:,self.d_num:]
            x_num=self.num_embeddings(x_num).flatten(1)
            candidate_x_num=self.num_embeddings(candidate_x_num).flatten(1)
            x=torch.cat([x_num,x_cat],dim=-1)
            candidate_x=torch.cat([candidate_x_num,candidate_x_cat],dim=-1)
        # x=x.double()
        # candidate_x=candidate_x.double()
        if self.n_blocks > 0:
            candidate_x =self.post_encoder(self.encoder(candidate_x))
            x = self.post_encoder(self.encoder(x))          
        else:         
            candidate_x = self.encoder(candidate_x)
            x = self.encoder(x)
        if is_train:
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None
        
        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y.long(), self.d_out).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y=candidate_y.unsqueeze(-1)

        # calculate distance
        # default we use euclidean distance, however, cosine distance is also a good choice for classification, after tuning
        # of temperature, cosine distance even outperforms euclidean distance for classification
        distances = torch.cdist(x, candidate_x, p=2)
        # x=F.normalize(x,p=2,dim=-1)   # this is code for cosine distance
        # candidate_x=F.normalize(candidate_x,p=2,dim=-1)
        # distances=torch.mm(x,candidate_x.T)
        # distances=-distances
        distances=distances/self.T
        # remove the label of training index
        if is_train:
            distances = distances.clone().fill_diagonal_(torch.inf)     
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        eps=1e-7
        if self.d_out>1:
            # if task type is classification, since the logit is already normalized, we calculate the log of the logit 
            # and use nll_loss to calculate the loss
            logits=torch.log(logits+eps)
        return logits.squeeze(-1)


def make_random_batches(
    train_size: int, batch_size: int, device: Optional[torch.device] = None
) :
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)
    # this function is borrowed from tabr
    # Below, we check that we do not face this issue:
    # https://github.com/pytorch/vision/issues/3816
    # This is still noticeably faster than running randperm on CPU.
    # UPDATE: after thousands of experiments, we faced the issue zero times,
    # so maybe we should remove the assert.
    assert torch.equal(
        torch.arange(train_size, device=device), permutation.sort().values
    )
    return batches  # type: ignore[code]

class ModernNCAMethod(object, metaclass=abc.ABCMeta):
    def __init__(self, params, tasktype, num_cols=[], cat_features=[], input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="modernnca"):
        super().__init__()

        self.num_cols = num_cols
        self.cat_features = cat_features
        self.tasktype = tasktype
        self.params = params
        self.data_id = data_id
        self.device = device
        self.max_epoch = 100

        self.is_binclass = (self.tasktype == "binclass")
        self.is_multiclass = (self.tasktype == "multiclass")
        self.is_regression = (self.tasktype == "regression")
        self.n_num_features = len(self.num_cols)
        self.n_cat_features = len(self.cat_features)

        self.train_step = 0
        
        self.model = ModernNCA(d_in = (len(num_cols) + len(cat_features)),
            d_num = len(num_cols),
            d_out = output_dim,
            **params["model"]).to(device)
        self.model.float()

        self._callback_container = CallbackContainer([EarlyStopping(
            early_stopping_metric="val_loss",
            patience=params["early_stopping_rounds"],
        )])

        self.trlog = {}
        self.trlog['args'] = params
        self.trlog['train_loss'] = []
        self.trlog['best_epoch'] = 0
        if self.is_regression:
            self.trlog['best_res'] = 1e10
        else:
            self.trlog['best_res'] = 0

    def fit(self, X_train, y_train, X_val, y_val):
        self.N = X_train[:, self.num_cols]
        self.C = X_train[:, self.cat_features]
        if self.tasktype == "multiclass":
            self.y = torch.argmax(y_train, dim=1)
        else:
            self.y = y_train

        self.batch_size = get_batch_size(len(X_train))

        if self.tasktype == "regression":
            self.criterion = F.mse_loss 
        elif self.tasktype == "multiclass":
            self.criterion = F.cross_entropy
        else:
            self.criterion = F.binary_cross_entropy
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.params['lr'], 
            weight_decay=self.params['weight_decay']
        )
        
        if self.params["lr_scheduler"] & (len(X_train) > self.batch_size):
            self.scheduler = CosineAnnealingLR_Warmup(self.optimizer, warmup_epochs=10, T_max=100, iter_per_epoch=len(X_train)//self.batch_size, 
                                                      base_lr=self.params['lr'], warmup_lr=1e-6, eta_min=0, last_epoch=-1)
            
        self.train_size = self.N.shape[0] if self.N is not None else self.C.shape[0]
        self.train_indices = torch.arange(min([self.train_size, 50000]), device=self.device)

        pbar = tqdm(range(1, self.max_epoch+1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            tic = time.time()
            loss = self.train_epoch(epoch)
            self.validate(epoch, X_val, y_val)
            elapsed = time.time() - tic
            pbar.set_postfix_str(f'Time cost: {elapsed}, Tr loss: {loss:.5f}')
            if not self.continue_training:
                break

    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        for batch_idx in make_random_batches(self.train_size, self.batch_size, self.device):
            self.train_step = self.train_step + 1
            
            X_num = self.N[batch_idx] if self.N is not None else None
            X_cat = self.C[batch_idx] if self.C is not None else None
            y = self.y[batch_idx]

            candidate_indices = self.train_indices
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, batch_idx)]

            candidate_x_num = self.N[candidate_indices] if self.N is not None else None
            candidate_x_cat = self.C[candidate_indices] if self.C is not None else None
            candidate_y = self.y[candidate_indices]
            X_num = X_num.float() if X_num is not None else None
            X_cat = X_cat.float()   if X_cat is not None else None
            candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
            if self.is_regression:
                candidate_y = candidate_y.float()
                y = y.float()
            if X_cat is None and X_num is not None:
                x, candidate_x = X_num, candidate_x_num
            elif X_cat is not None and X_num is None:
                x, candidate_x = X_cat, candidate_x_cat
            else:
                x, candidate_x = torch.cat([X_num, X_cat], dim=1),torch.cat([candidate_x_num, candidate_x_cat], dim=1)

            if x.size(0) > 1:
                pred = self.model(
                    x=x,
                    y=y,
                    candidate_x=candidate_x,
                    candidate_y=candidate_y,
                    is_train=True,
                ).squeeze(-1)

                # if self.tasktype == "binclass":
                    # pred = torch.clamp(pred, min=1e-10, max=1-1e-10)
                loss = self.criterion(pred, y)
                tl.add(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.params["lr_scheduler"] & (len(self.y) > self.batch_size):
                    self.scheduler.step()

        tl = tl.item()
        self.trlog['train_loss'].append(tl)

        return tl


    def validate(self, epoch, X_val, y_val):
        if self.tasktype == "multiclass":
            y_val = torch.argmax(y_val, dim=1)
        else:
            y_val = y_val
            
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            candidate_x_num = self.N[:50000] if self.N is not None else None
            candidate_x_cat = self.C[:50000] if self.C is not None else None
            candidate_y = self.y[:50000]
            candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
            if self.is_regression:
                candidate_y = candidate_y.float()

            logits = []
            iters = X_val.shape[0] // 10000 + 1
            for i in range(iters):
                N = X_val[10000*i:10000*(i+1), self.num_cols]
                C = X_val[10000*i:10000*(i+1), self.cat_features]
                if len(self.num_cols) == 0:
                    X_num, X_cat = None, C
                elif len(self.cat_features) == 0:
                    X_num, X_cat = N, None
                else:
                    X_num, X_cat = N, C
                
                X_num = X_num.float() if X_num is not None else None
                X_cat = X_cat.float() if X_cat is not None else None
    
                if X_cat is None and X_num is not None:
                    x, candidate_x = X_num, candidate_x_num
                elif X_cat is not None and X_num is None:
                    x, candidate_x = X_cat, candidate_x_cat
                else:
                    x, candidate_x = torch.cat([X_num, X_cat], dim=1),torch.cat([candidate_x_num, candidate_x_cat], dim=1)
    
                val_pred = self.model(
                    x = x,
                    y = None,
                    candidate_x = candidate_x,
                    candidate_y = candidate_y,
                    is_train = False,
                ).squeeze(-1)

                # if self.tasktype == "binclass":
                    # val_pred = torch.clamp(val_pred, min=1e-10, max=1-1e-10)

                logits.append(val_pred)

            logits = torch.concatenate(logits, dim=0)
            val_loss = self.criterion(logits, y_val).item()
        
        self._callback_container.on_epoch_end(epoch, {"val_loss": val_loss, "epoch": epoch})
        if any([cb.should_stop for cb in self._callback_container.callbacks]):
            self.continue_training = False
        else:
            self.continue_training = True
                
    def predict(self, X_test):        
        self.model.eval()
        with torch.no_grad():
            candidate_x_num = self.N[:50000] if self.N is not None else None
            candidate_x_cat = self.C[:50000] if self.C is not None else None
            candidate_y = self.y[:50000]
            candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
            if self.is_regression:
                candidate_y = candidate_y.float()

            logits = []
            iters = X_test.shape[0] // 10000 + 1
            for i in range(iters):
                N = X_test[10000*i:10000*(i+1), self.num_cols]
                C = X_test[10000*i:10000*(i+1), self.cat_features]
                if len(self.num_cols) == 0:
                    X_num, X_cat = None, C
                elif len(self.cat_features) == 0:
                    X_num, X_cat = N, None
                else:
                    X_num, X_cat = N, C
                
                X_num = X_num.float() if X_num is not None else None
                X_cat = X_cat.float() if X_cat is not None else None
    
                if X_cat is None and X_num is not None:
                    x, candidate_x = X_num, candidate_x_num
                elif X_cat is not None and X_num is None:
                    x, candidate_x = X_cat, candidate_x_cat
                else:
                    x, candidate_x = torch.cat([X_num, X_cat], dim=1),torch.cat([candidate_x_num, candidate_x_cat], dim=1)
    
                val_pred = self.model(
                    x = x,
                    y = None,
                    candidate_x = candidate_x,
                    candidate_y = candidate_y,
                    is_train = False,
                ).squeeze(-1)

                # if self.tasktype == "binclass":
                    # val_pred = torch.clamp(val_pred, min=1e-10, max=1-1e-10)

                logits.append(val_pred)

            logits = torch.concatenate(logits, dim=0)

        if self.tasktype == "binclass":
            return torch.round(logits).detach().cpu().numpy()
        elif self.tasktype == "regression":
            return logits.detach().cpu().numpy()
        else:
            return torch.argmax(logits, dim=1).detach().cpu().numpy()

    def predict_proba(self, X_test, logit=False):
        self.model.eval()
        with torch.no_grad():
            candidate_x_num = self.N[:50000] if self.N is not None else None
            candidate_x_cat = self.C[:50000] if self.C is not None else None
            candidate_y = self.y[:50000]
            candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
            if self.is_regression:
                candidate_y = candidate_y.float()

            logits = []
            iters = X_test.shape[0] // 10000 + 1
            for i in range(iters):
                N = X_test[10000*i:10000*(i+1), self.num_cols]
                C = X_test[10000*i:10000*(i+1), self.cat_features]
                if len(self.num_cols) == 0:
                    X_num, X_cat = None, C
                elif len(self.cat_features) == 0:
                    X_num, X_cat = N, None
                else:
                    X_num, X_cat = N, C
                
                X_num = X_num.float() if X_num is not None else None
                X_cat = X_cat.float() if X_cat is not None else None
    
                if X_cat is None and X_num is not None:
                    x, candidate_x = X_num, candidate_x_num
                elif X_cat is not None and X_num is None:
                    x, candidate_x = X_cat, candidate_x_cat
                else:
                    x, candidate_x = torch.cat([X_num, X_cat], dim=1),torch.cat([candidate_x_num, candidate_x_cat], dim=1)
    
                val_pred = self.model(
                    x = x,
                    y = None,
                    candidate_x = candidate_x,
                    candidate_y = candidate_y,
                    is_train = False,
                ).squeeze(-1)

                # if self.tasktype == "binclass":
                #     val_pred = torch.clamp(val_pred, min=1e-10, max=1-1e-10)
                    
                logits.append(val_pred)

            logits = torch.concatenate(logits, dim=0)

        if logit:
            return logits.detach().cpu().numpy()
        else:
            return torch.nn.functional.softmax(logits).detach().cpu().numpy()