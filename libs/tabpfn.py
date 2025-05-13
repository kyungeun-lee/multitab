import torch
import numpy as np
from tabpfn import TabPFNClassifier

class tabpfn(torch.nn.Module):
    def __init__(self, tasktype):
        
        super(tabpfn, self).__init__()
        self.tasktype = tasktype
        self.model = TabPFNClassifier(device=torch.device("cpu"), N_ensemble_configurations=32)
    
    def fit(self, X_train, y_train, X_val, y_val):
        if self.tasktype == "multiclass":
            y_train = torch.argmax(y_train, dim=1)
        self.model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
            
    def predict(self, X_test):
        return self.model.predict(X_test.cpu().numpy())
        
    def predict_proba(self, X_test, logit=False):
        return self.model.predict_proba(X_test.cpu().numpy())
