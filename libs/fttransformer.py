from libs.supervised import supmodel
import torch, math, typing

## Reference
## Revisiting deep learning models for tabular data (Y Gorishniy et al., NeurIPS 2021)
## https://github.com/yandex-research/rtdl
## source: https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py

class Tokenizer(torch.nn.Module):
    
    def __init__(
        self,
        d_numerical,
        categories,
        d_token,
        bias
    ):
        super().__init__()
        if len(categories) == 0:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            categories = categories.detach().numpy().tolist()
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            assert len(category_offsets) == len(categories)
            self.category_embeddings = torch.nn.Embedding(sum(categories), d_token)
            torch.nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
#             print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = torch.nn.Parameter(torch.Tensor(d_numerical + 1, d_token))
        self.bias = torch.nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias != None:
            torch.nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets == None else len(self.category_offsets))

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat == None else x_cat
        assert x_some != None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]
            + ([] if x_num == None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        
        if x_cat != None:
            emb = x_cat + self.category_offsets[None]
            x = torch.cat(
                [x, self.category_embeddings(emb.to(torch.long))],
                dim=1,
            )
        if self.bias != None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(torch.nn.Module):
    def __init__(
        self, d, n_heads, dropout, initialization
    ):
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = torch.nn.Linear(d, d)
        self.W_k = torch.nn.Linear(d, d)
        self.W_v = torch.nn.Linear(d, d)
        self.W_out = torch.nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                torch.nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            torch.nn.init.zeros_(m.bias)
        if self.W_out != None:
            torch.nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head).transpose(1, 2).reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q,
        x_kv,
        key_compression,
        value_compression,
    ):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression != None:
            assert value_compression != None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = torch.nn.functional.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value).transpose(1, 2).reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out != None:
            x = self.W_out(x)
        return x
    
    
def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * torch.nn.functional.relu(b)

def geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * torch.nn.functional.gelu(b)
    
class Transformer(torch.nn.Module):
    def __init__(self, params, num_cols=[], categories=[], input_dim=0, output_dim=0, device='cuda'):
        assert (params["kv_compression"] == None) ^ (params["kv_compression_sharing"] != None)
        
        super(Transformer, self).__init__()
        d_numerical = len(num_cols)
        self.x_cat = [i for i in range(input_dim) if i not in num_cols]
        self.x_categories = categories
        self.x_num = num_cols
        self.d_token = params["d_token"] * params["n_heads"]
        self.tokenizer = Tokenizer(d_numerical, categories, self.d_token, params["token_bias"])
        n_tokens = self.tokenizer.n_tokens
        
        self.optimizer = params["optimizer"]
        self.learning_rate = params["learning_rate"]
        self.weight_decay = params["weight_decay"]
        
        def make_kv_compression():
            assert kv_compression
            compression = torch.nn.Linear(
                n_tokens, int(n_tokens * params["kv_compression"]), bias=False
            )
            if params["initialization"] == 'xavier':
                torch.nn.init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if params["kv_compression"] and params["kv_compression_sharing"] == 'layerwise'
            else None
        )

        def make_normalization():
            return torch.nn.LayerNorm(self.d_token)
        
        d_hidden = int(self.d_token * params["d_ffn_factor"])
        self.layers = torch.nn.ModuleList([])
        for layer_idx in range(params["n_layers"]):
            layer = torch.nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        self.d_token, params["n_heads"], params["attention_dropout"], params["initialization"]
                    ),
                    'linear0': torch.nn.Linear(
                        self.d_token, d_hidden * (2 if params["activation"].endswith('glu') else 1)
                    ),
                    'linear1': torch.nn.Linear(d_hidden, self.d_token),
                    'norm1': make_normalization(),
                }
            )
            if not params["prenormalization"] or layer_idx:
                layer['norm0'] = make_normalization()
            if params["kv_compression"] and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if params["kv_compression_sharing"] == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert params["kv_compression_sharing"] == 'key-value'
            self.layers.append(layer)

        if params["activation"] == "reglu":
            self.activation = reglu
            self.last_activation = torch.nn.functional.relu
        elif params["activation"] == "geglu":
            self.activation = geglu
            self.last_activation = torch.nn.functional.gelu
        elif params["activation"] == "sigmoid":
            self.activation = torch.sigmoid
            self.last_activation = torch.sigmoid
        elif params["activation"] == "relu":
            self.activation = torch.nn.functional.relu
            self.last_activation = torch.nn.functional.relu
        else:
            raise ValueError

        self.prenormalization = params["prenormalization"]
        self.last_normalization = make_normalization() if params["prenormalization"] else None
        self.ffn_dropout = params["ffn_dropout"]
        self.residual_dropout = params["residual_dropout"]
        self.head = torch.nn.Linear(self.d_token, output_dim)
        
    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = torch.nn.functional.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_all, cat_features=[]):
        x_num = x_all[:, self.x_num] if len(self.x_num) > 0 else None
        x_cat = x_all[:, self.x_cat] if len(self.x_cat) > 0 else None

        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = typing.cast(typing.Dict[str, torch.nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = torch.nn.functional.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
        
        
class build_ftt(Transformer):
    def __init__(self, params, num_cols=[], categories=[], input_dim=0, output_dim=0, device='cuda'):
        super().__init__(params, num_cols, categories, input_dim, output_dim, device)
        self.model = Transformer(params, num_cols, categories, input_dim, output_dim, device)
    
    def forward(self, x, cat_features=[]):
        return self.model(x)
    
    def make_optimizer(self):
        def needs_wd(name):
            return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

        for x in ['tokenizer', '.norm', '.bias']:
            assert any(x in a for a in (b[0] for b in self.model.named_parameters()))
        parameters_with_wd = [v for k, v in self.model.named_parameters() if needs_wd(k)]
        parameters_without_wd = [v for k, v in self.model.named_parameters() if not needs_wd(k)]

        parameter_groups = ([{'params': parameters_with_wd}, {'params': parameters_without_wd, 'weight_decay': 0.0}])

        if self.optimizer == "AdamW":
            return torch.optim.AdamW(parameter_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(parameter_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(parameter_groups, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)

class FTTransformer(supmodel):
    def __init__(self, params, tasktype, num_cols=[], categories=[], input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="ftt"):
        
        super().__init__(tasktype, params, device, data_id, modelname)
        self.model = build_ftt(params, num_cols, categories, input_dim, output_dim, device)
        self.model = self.model.to(device)
    
    def fit(self, X_train, y_train):
        
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0])]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train)]
            y_train = y_train[~torch.isnan(y_train)]
        
        ### if we use early stopping!
        n_samples = len(X_train)
        train_idx = np.random.choice(n_samples, int(0.9*n_samples), replace=False)
        X_val = X_train[~train_idx]
        y_val = y_train[~train_idx]
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_val = y_val.unsqueeze(1)
        
        super().fit(X_train, y_train, X_val, y_val)
            
    def predict(self, X_test):
        return super().predict(X_test)
        
    def predict_proba(self, X_test, logit=False):
        return super().predict_proba(X_test, logit=logit)
