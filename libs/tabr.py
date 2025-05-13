import torch
import torch.nn as nn
import torch.nn.functional as F

import faiss
import delu
from scipy.special import expit

import abc
import math, time
import statistics
from functools import partial
from typing import Any, Callable, Optional, Union, cast
import typing as ty
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


class TabR(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_cat_features: int,
        n_classes: Optional[int],
        num_embeddings: Optional[dict],
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization,
        context_dropout: float,
        dropout0: float,
        dropout1,
        normalization: str,
        activation: str,
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        if dropout1 == 'dropout0':
            dropout1 = dropout0
        self.n_classes = n_classes

        self.num_embeddings = (
            None
            if num_embeddings is None
            else PLREmbeddings(**num_embeddings, n_features=n_num_features)
        )

        self.n_num_features = n_num_features
        d_in = (
            n_num_features * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_cat_features
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList([make_block(i > 0) for i in range(encoder_n_blocks)])

        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes == 1
            else nn.Sequential(
                nn.Embedding(n_classes, d_main),
                delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        self.blocks1 = nn.ModuleList([make_block(True) for _ in range(predictor_n_blocks)])
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, n_classes),
        )

        self.search_index = None
        self.memory_efficient = False
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

        # Cached candidates
        self.cached_candidate_k = None
        self.cached_candidate_y = None

    def update_index(self):
        """Update FAISS search index once per epoch."""
        if self.cached_candidate_k is None or self.cached_candidate_y is None:
            return
        d_main = self.cached_candidate_k.shape[1]
        if self.search_index is None:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = torch.cuda.current_device()
            self.search_index = faiss.GpuIndexFlatL2(res, d_main, cfg)
        self.search_index.reset()
        self.search_index.add(self.cached_candidate_k.to(torch.float32).detach().cpu().numpy())

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)
        else:
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)

    def _encode(self, x_num, x_cat):
        x = []
        if x_num is None:
            self.num_embeddings = None
        else:
            x.append(
                x_num if self.num_embeddings is None else self.num_embeddings(x_num).flatten(1)
            )
        if x_cat is not None:
            x.append(x_cat)
        x = torch.cat(x, dim=1)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
        self,
        *,
        x_num: Tensor,
        x_cat: ty.Optional[Tensor],
        y: Optional[Tensor],
        candidate_x_num: ty.Optional[Tensor],
        candidate_x_cat: ty.Optional[Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:

        device = x_num.device if x_num is not None else x_cat.device

        if self.cached_candidate_k is None:
            with torch.no_grad():
                self.cached_candidate_k = (
                    self._encode(candidate_x_num, candidate_x_cat)[1]
                    if self.candidate_encoding_batch_size is None
                    else torch.cat([
                        self._encode(xn, xc)[1]
                        for xn, xc in delu.iter_batches(
                            (candidate_x_num, candidate_x_cat), self.candidate_encoding_batch_size
                        )
                    ])
                )
                self.cached_candidate_y = candidate_y

        x, k = self._encode(x_num, x_cat)
        if is_train:
            assert y is not None
            candidate_k = torch.cat([k, self.cached_candidate_k])
            candidate_y = torch.cat([y, self.cached_candidate_y])
        else:
            candidate_k = self.cached_candidate_k
            candidate_y = self.cached_candidate_y

        batch_size, d_main = k.shape

        with torch.no_grad():
            distances, context_idx = self.search_index.search(
                k.to(torch.float32).detach().cpu().numpy(), context_size + (1 if is_train else 0)
            )
            distances = torch.tensor(distances, device=device)
            context_idx = torch.tensor(context_idx, device=device)
            if is_train:
                distances[context_idx == torch.arange(batch_size, device=device)[:, None]] = torch.inf
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        context_k = candidate_k[context_idx]

        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        if self.n_classes > 1:
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None].long())
        else:
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
            if len(context_y_emb.shape) == 4:
                context_y_emb = context_y_emb[:, :, 0, :]

        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x

class TabRMethod(object, metaclass=abc.ABCMeta):
    def __init__(self, params, tasktype, num_cols=[], cat_features=[], input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="tabr"):
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
        self.context_size = 96

        self.model = TabR(
            n_num_features = self.n_num_features,
            n_cat_features = self.n_cat_features,
            n_classes = output_dim,
            **params["model"]
        ).to(device)
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
            self.criterion = F.binary_cross_entropy_with_logits
            
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

        if self.model.cached_candidate_k is None:
            candidate_x_num = self.N[:50000].float().to(self.device) if self.N is not None else None
            candidate_x_cat = self.C[:50000].float().to(self.device) if self.C is not None else None
            candidate_y = self.y[:50000].float().to(self.device) if self.is_regression else self.y[:50000].to(self.device)
            with torch.no_grad():
                self.model.cached_candidate_k = self.model._encode(candidate_x_num, candidate_x_cat)[1]
                self.model.cached_candidate_y = candidate_y

        self.model.update_index()

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
                    x_num=x[:,:self.n_num_features], x_cat=x[:,self.n_num_features:], y=y, 
                    candidate_x_num=candidate_x_num,
                    candidate_x_cat=candidate_x_cat,
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=True,
                ).squeeze(-1)

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
                    x_num=x[:,:self.n_num_features], x_cat=x[:,self.n_num_features:], y=None, 
                    candidate_x_num=candidate_x[:,:self.n_num_features],
                    candidate_x_cat=candidate_x[:,self.n_num_features:],
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)

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
                    x_num=x[:,:self.n_num_features], x_cat=x[:,self.n_num_features:], y=None, 
                    candidate_x_num=candidate_x[:,:self.n_num_features],
                    candidate_x_cat=candidate_x[:,self.n_num_features:],
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)

                logits.append(val_pred)

            logits = torch.concatenate(logits, dim=0)

        if self.tasktype == "binclass":
            return torch.round(torch.sigmoid(logits)).detach().cpu().numpy()
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
                    x_num=x[:,:self.n_num_features], x_cat=x[:,self.n_num_features:], y=None, 
                    candidate_x_num=candidate_x[:,:self.n_num_features],
                    candidate_x_cat=candidate_x[:,self.n_num_features:],
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)
                    
                logits.append(val_pred)

            logits = torch.concatenate(logits, dim=0)

        if logit:
            return logits.detach().cpu().numpy()
        else:
            return torch.nn.functional.softmax(logits).detach().cpu().numpy()