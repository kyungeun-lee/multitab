from libs.supervised import supmodel
import torch, math, typing

def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * torch.nn.functional.relu(b)

def geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * torch.nn.functional.gelu(b)

class resnet(torch.nn.Module):
    def __init__(
        self,
        num_cols,
        categories,
        d_embedding,
        d,
        d_hidden_factor,
        n_layers,
        activation,
        normalization,
        hidden_dropout,
        residual_dropout,
        d_out
    ):
        super().__init__()
        d_numerical = len(num_cols)
        input_dim = d_numerical + len(categories)
        self.x_cat = [i for i in range(input_dim) if i not in num_cols]
        self.x_categories = categories
        self.x_num = num_cols
            
        def make_normalization():
            return {'batchnorm': torch.nn.BatchNorm1d, 'layernorm': torch.nn.LayerNorm}[
                normalization
            ](d)
        
        if activation == "reglu":
            self.main_activation = reglu
            self.last_activation = torch.nn.functional.relu
        elif activation == "geglu":
            self.main_activation = geglu
            self.last_activation = torch.nn.functional.gelu
        elif activation == "sigmoid":
            self.main_activation = torch.sigmoid
            self.last_activation = torch.sigmoid
        elif activation == "relu":
            self.main_activation = torch.nn.functional.relu
            self.last_activation = torch.nn.functional.relu
        else:
            raise ValueError
        
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if len(categories) > 0:
            categories = categories.detach().numpy().tolist()
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = torch.nn.Embedding(sum(categories), d_embedding)
            torch.nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
#             print(f'{self.category_embeddings.weight.shape=}')
        
        self.first_layer = torch.nn.Linear(d_in, d)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': torch.nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': torch.nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = torch.nn.Linear(d, d_out)

    def forward(self, x_all):
        x_num = x_all[:, self.x_num] if len(self.x_num) > 0 else None
        x_cat = x_all[:, self.x_cat] if len(self.x_cat) > 0 else None

        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat != None:
            emb = x_cat + self.category_offsets[None]
            x.append(self.category_embeddings(emb.to(torch.long)).view(x_cat.size(0), -1))
        x = torch.cat(x, dim=-1)
        
#         import IPython; IPython.embed()
        x = self.first_layer(x)
        for layer in self.layers:
            layer = typing.cast(typing.Dict[str, torch.nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = torch.nn.functional.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = torch.nn.functional.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

class build_resnet(resnet):
    def __init__(self, 
                 num_cols, categories, d_embedding, d, d_hidden_factor, n_layers, activation, normalization, hidden_dropout, residual_dropout, d_out,
                 optimizer, learning_rate, weight_decay):
        
        super().__init__(num_cols, categories, d_embedding, d, d_hidden_factor, n_layers, activation, normalization, hidden_dropout, residual_dropout, d_out)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = resnet(num_cols, categories, d_embedding, d, d_hidden_factor, n_layers, activation, normalization, hidden_dropout, residual_dropout, d_out)
    
    def forward(self, x, cat_features=[]):
        return self.model(x)
    
    def make_optimizer(self):
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)

    
class ResNet(supmodel):
    def __init__(self, params, num_cols=[], cat_features=[], input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="resnet"):
        
        super().__init__(params, device, data_id, modelname)
        self.model = build_resnet(num_cols, cat_features, params["d_embedding"], params["d"], params["d_hidden_factor"], 
                                  params["n_layers"], params["activation"], params["normalization"], params["hidden_dropout"], params["residual_dropout"], output_dim,
                                  params["optimizer"], params["learning_rate"], params["weight_decay"])
        self.model = self.model.to(device)
    
    def fit(self, X_train, y_train, X_val, y_val):
        
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0])]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train)]
            y_train = y_train[~torch.isnan(y_train)]
            
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_val = y_val.unsqueeze(1)
        
        super().fit(X_train, y_train, X_val, y_val)
            
    def predict(self, X_test):
        return super().predict(X_test)
        
    def predict_proba(self, X_test, logit=False):
        return super().predict_proba(X_test, logit=logit)