import torch, logging
from tqdm import tqdm
from libs.data import get_batch_size
import numpy as np

class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

class CallbackContainer:
    def __init__(self, callbacks):
        self.callbacks = callbacks or []

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_batch_begin(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs)
            
class EarlyStopping:
    def __init__(self, early_stopping_metric="val_loss", patience=500):
        self.early_stopping_metric = early_stopping_metric
        self.patience = patience
        self.best_value = None
        self.patience_counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.early_stopping_metric)
        if current_value is None:
            return

        if self.best_value is None:
            self.best_value = current_value

        if current_value < self.best_value:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.should_stop = True
            
class supmodel(torch.nn.Module):
    def __init__(self, tasktype, params, device, data_id=None, modelname=None, cat_features=[]):
        
        super(supmodel, self).__init__()
        
        self.cat_features = cat_features
        self.device = device
        self.params = params
        self.data_id = data_id
        self.modelname = modelname
        self.tasktype = tasktype
        
        #reference: TabZilla
        self._callback_container = CallbackContainer([EarlyStopping(
            early_stopping_metric="val_loss",
            patience=params["early_stopping_rounds"],
        )])
    
    def fit(self, X_train, y_train, X_val, y_val):
            
        batch_size = get_batch_size(len(X_train))
            
        optimizer = self.model.make_optimizer()
        if self.tasktype == "regression":
            loss_fn = torch.nn.functional.mse_loss
        elif self.tasktype == "binclass":
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            loss_fn = torch.nn.functional.cross_entropy
            
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
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
            
            for i, (x, y) in enumerate(train_loader):
                self.model.train(); optimizer.zero_grad()
                
                out = self.model(x.to(self.device), self.cat_features)
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
            # import pdb; pdb.set_trace();
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    out = self.model(x_val.to(self.device), self.cat_features)
                    if out.size() != y_val.size():
                        out = out.view(y_val.size())
                    val_loss += loss_fn(out, y_val.to(self.device)).item()
            val_loss /= len(val_loader)
            
            self._callback_container.on_epoch_end(epoch, {"val_loss": val_loss, "epoch": epoch})
            if any([cb.should_stop for cb in self._callback_container.callbacks]):
                print(f"Early stopping at epoch {epoch}")
                break
                
        self.model.eval()
    
    def predict(self, X_test, cat_features=[]):
        with torch.no_grad():
            if (X_test.shape[0] > 10000):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    inputs = X_test[100*i:100*(i+1)]
                    if inputs.size(0) > 0:
                        pred = self.model(inputs, cat_features)
                        logits.append(pred)
                        del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.model(X_test, cat_features)

            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().detach().cpu().numpy()
            elif self.tasktype == "regression":
                return logits.detach().cpu().numpy()
            else:
                return torch.argmax(logits, dim=1).detach().cpu().numpy()
    
    def predict_proba(self, X_test, cat_features=[], logit=False):
        with torch.no_grad():
            if (X_test.shape[0] > 10000) or (X_test.shape[1] > 240):
                logits = []
                iters = X_test.shape[0] // 100 + 1
                for i in range(iters):
                    inputs = X_test[100*i:100*(i+1)]
                    if inputs.size(0) > 0:
                        pred = self.model(inputs, cat_features)
                        logits.append(pred)
                        del pred
                logits = torch.concatenate(logits, dim=0)
            else:
                logits = self.model(X_test, cat_features)

            if logit:
                return logits.detach().cpu().numpy()
            else:
                return torch.nn.functional.softmax(logits).detach().cpu().numpy()
    
    
    
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