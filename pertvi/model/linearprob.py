import logging
import warnings
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import math
import torch
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from scvi import settings

from torch.utils.data import DataLoader, TensorDataset

class LPModel():
    def __init__(
        self,
        adata: AnnData,
        x: str,
        y: str,
        label: Optional[str] = None,
        x2_label: Optional[str] = None,
        mask_label: Optional[str] = None,
        mode: str = 'Weighted',
        norm: str = 'none',
        hidden_dim = 10,
    ):
        available_method_kinds = ["Weighted", "WeightedDisentangled", "Contrast", "Individual"]
        assert mode in available_method_kinds, (
            f"mode = {mode} is not one of"
            f" {available_method_kinds}"
        )
        self.mode = mode
        self.norm = norm
        self.x_label = x
        self.y_label = y
        self.ind_label = label
        self.x2_label = x2_label
        self.y_index = pd.get_dummies(adata.obs[y]).columns
        if label is not None:
            self.labels = pd.get_dummies(adata.obs[label]).values.argmax(axis=1)
            
        if mask_label is None:
            self.mask_label = self.ind_label
            self.mask_labels = self.labels
        else:
            self.mask_label = mask_label
            self.mask_labels = pd.get_dummies(adata.obs[mask_label]).values.argmax(axis=1)
        
        if mode == 'Weighted':
            self.X = adata.obsm[x]
            self.Y = pd.get_dummies(adata.obs[y]).values
            self.module = WeightedLinearProbing(self.X.shape[1],self.Y.shape[1],hidden_dim, norm)
        elif mode == 'WeightedDisentangled':
            self.X1 = adata.obsm[x]
            self.X2 = adata.obsm[x2_label]
            self.Y = pd.get_dummies(adata.obs[y]).values
            self.module = WeightedDisentangledLinearProbing(self.X1.shape[1],self.X2.shape[1],self.Y.shape[1],hidden_dim, norm)
        elif mode == 'Contrast':
            self.X = adata.obsm[x]
            self.Y = pd.get_dummies(adata.obs[y]).values
            self.labels2 = pd.get_dummies(adata.obs[x2_label]).values.argmax(axis=1)
            self.module = ContrastLinearProbing(self.X.shape[1],self.Y.shape[1],hidden_dim, norm)    
        else:
            self.X = adata.obsm[x]
            self.Y = pd.get_dummies(adata.obs[y]).values
            self.module = LinearProbing(self.X.shape[1],self.Y.shape[1], norm)
            
    
    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        batch_size: Optional[int] = None,
        validation_size: Optional[float] = None,
        lr = 1e-3,
        weight_decay = 1e-4,
        device = None,
        seed = None,
    ):
        n_cells = self.Y.shape[0]
        if max_epochs is None:
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
            

        if validation_size is None:
            validation_size = 1 - train_size


        if device is None:
            if use_gpu & torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = 'cpu'
        self.module = self.module.to(device)
        self.device = device
        
        if seed is None:
            seed = settings.seed
        
        if self.mask_label is not None:
            #mask_labels = pd.get_dummies(adata.obs[mask_label]).values.argmax(axis=1)
            unique_label = np.unique(self.mask_labels)
            n_label = unique_label.shape[0]
            n_train = int(train_size *n_label)
            n_val = int(validation_size *n_label)
            random_state = np.random.RandomState(seed=seed)
            permutation = random_state.permutation(n_label)
            self.train_mask = np.isin(self.mask_labels,unique_label[permutation][:n_train])
            self.val_mask = np.isin(self.mask_labels, unique_label[permutation][n_train : (n_val + n_train)])
            self.test_mask = np.isin(self.mask_labels, unique_label[permutation][(n_val + n_train):])
        else:
            n_train = int(train_size *n_cells)
            n_val = int(validation_size *n_cells)
            random_state = np.random.RandomState(seed=seed)
            permutation = random_state.permutation(n_cells)
            self.train_mask = permutation[:n_train]
            self.val_mask = permutation[n_train : (n_val + n_train)]
            self.test_mask = permutation[(n_val + n_train):]
            
        
        if self.mode == 'Weighted':
            self.X = torch.Tensor(self.X)
            self.Y = torch.Tensor(self.Y)
            self.labels = torch.Tensor(self.labels)
            train_dataset = TensorDataset(self.X[self.train_mask], self.Y[self.train_mask], self.labels[self.train_mask])
            val_dataset = TensorDataset(self.X[self.val_mask], self.Y[self.val_mask], self.labels[self.val_mask])
            
        elif self.mode == 'Contrast':
            self.X = torch.Tensor(self.X)
            self.Y = torch.Tensor(self.Y)
            self.labels = torch.Tensor(self.labels)
            self.labels2 = torch.Tensor(self.labels2)
            train_dataset = TensorDataset(self.X[self.train_mask], self.Y[self.train_mask], self.labels[self.train_mask], self.labels2[self.train_mask])
            val_dataset = TensorDataset(self.X[self.val_mask], self.Y[self.val_mask], self.labels[self.val_mask], self.labels2[self.val_mask])
        
        elif self.mode == 'WeightedDisentangled':
            self.X1 = torch.Tensor(self.X1)
            self.X2 = torch.Tensor(self.X2)
            self.Y = torch.Tensor(self.Y)
            self.labels = torch.Tensor(self.labels)
            train_dataset = TensorDataset(self.X1[self.train_mask],self.X2[self.train_mask], self.Y[self.train_mask], self.labels[self.train_mask])
            val_dataset = TensorDataset(self.X1[self.val_mask],self.X2[self.val_mask], self.Y[self.val_mask], self.labels[self.val_mask])
        else:

            self.X = torch.Tensor(self.X)
            self.Y = torch.Tensor(self.Y)
            train_dataset = TensorDataset(self.X[self.train_mask], self.Y[self.train_mask])
            val_dataset = TensorDataset(self.X[self.val_mask], self.Y[self.val_mask])
            
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if validation_size > 0:
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss = []
        val_loss = []
        pbar = tqdm(range(1, max_epochs + 1))

        for epoch in pbar:
            train_loss.append(_train(self.module, train_dataloader, self.mode,device,optimizer))
            pbar.set_description('Epoch '+str(epoch)+'/'+str(max_epochs))
            if validation_size > 0:
                val_loss.append(_eval(self.module, val_dataloader, self.mode,device))
                pbar.set_postfix(train_loss=train_loss[epoch-1], val_loss=val_loss[epoch-1])
            else:
                pbar.set_postfix(train_loss=train_loss[epoch-1])

        return train_loss, val_loss
    
    @torch.no_grad()
    def predict(
        self,
        adata: AnnData,
    ):
        self.module.eval()
        if self.mode == 'Weighted':
            X = torch.Tensor(adata.obsm[self.x_label]).to(self.device)
            labels = torch.Tensor(pd.get_dummies(adata.obs[self.ind_label]).values.argmax(axis=1)).to(self.device)
            predictions, unique_labels = self.module.predict(X,labels)
            indices = unique_labels.to(device='cpu', dtype=torch.int64).numpy()
            mapped_categories = pd.get_dummies(adata.obs[self.ind_label]).columns[indices]
            return predictions.cpu().numpy(), mapped_categories
        
        elif self.mode == 'Contrast':
            X = torch.Tensor(adata.obsm[self.x_label]).to(self.device)
            labels = torch.Tensor(pd.get_dummies(adata.obs[self.ind_label]).values.argmax(axis=1)).to(self.device)
            labels2 = torch.Tensor(pd.get_dummies(adata.obs[self.x2_label]).values.argmax(axis=1)).to(self.device)
            predictions, unique_labels = self.module.predict(X,labels,labels2)
            indices = unique_labels.to(device='cpu', dtype=torch.int64).numpy()
            mapped_categories = pd.get_dummies(adata.obs[self.ind_label]).columns[indices]
            return predictions.cpu().numpy(), mapped_categories
            
        elif self.mode == 'WeightedDisentangled':
            X1 = torch.Tensor(adata.obsm[self.x_label]).to(self.device)
            X2 = torch.Tensor(adata.obsm[self.x2_label]).to(self.device)
            labels = torch.Tensor(pd.get_dummies(adata.obs[self.ind_label]).values.argmax(axis=1)).to(self.device)
            predictions, unique_labels = self.module.predict(X1,X2,labels)
            indices = unique_labels.to(device='cpu', dtype=torch.int64).numpy()
            mapped_categories = pd.get_dummies(adata.obs[self.ind_label]).columns[indices]
            return predictions.cpu().numpy(), mapped_categories
            
        else:
            X = torch.Tensor(adata.obsm[self.x_label]).to(self.device)
            return self.module.predict(X).cpu().numpy()

def _train(model, dataloader, mode, device,optimizer):
    model.train()
    train_loss = []
    if mode == 'Weighted':
        for i, (x,y,label) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss = model(x,label,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu())
            
    elif mode == 'Contrast':
        for i, (x,y,label,label2) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            label = label.to(device)
            label2 = label2.to(device)
            optimizer.zero_grad()
            loss = model(x,label,label2,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu())        
            
    elif mode == 'WeightedDisentangled':
        for i, (x1,x2,y,label) in enumerate(dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss = model(x1,x2,label,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu())
    else:
        for i, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = model(x,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu())        
    return np.array(train_loss).mean()

def _eval(model, dataloader, mode, device):
    model.eval()
    val_loss = []
    if mode == 'Weighted':
        for i, (x,y,label,label2) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            label = label.to(device)
            label2 = label2.to(device)
            loss = model(x,label,label2,y)
            val_loss.append(loss.detach().cpu())
    elif mode == 'Contrast':
        for i, (x,y,label) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            label = label.to(device)
            loss = model(x,label,y)
            val_loss.append(loss.detach().cpu())
    elif mode == 'WeightedDisentangled':
        for i, (x1,x2,y,label) in enumerate(dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            label = label.to(device)
            loss = model(x1,x2,label,y)
            val_loss.append(loss.detach().cpu())
    else:
        for i, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            loss = model(x,y)
            val_loss.append(loss.detach().cpu())        
    return np.array(val_loss).mean()





class WeightedLinearProbing(nn.Module):
    def __init__(self, input_dim, y_dim, hidden_dim, use_norm = 'none'):
        super(WeightedLinearProbing, self).__init__()
        self.use_norm = use_norm
        if self.use_norm == 'batch':
            self.norm = nn.BatchNorm1d(input_dim)
        elif self.use_norm == 'layer':
            self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, 1)  # output the weights for each cell
        #self.linear1 = nn.Linear(hidden_dim, 1)
        self.linear2 = nn.Linear(input_dim, y_dim)  # output the weights for each cell
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels, y):
        if self.use_norm != 'none':
            x = self.norm(x)
        a = torch.sigmoid((self.linear(x))) * x
        #a = x
        unique, labels_tmp = torch.unique(labels, sorted=True, return_inverse=True)
        label = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, a.size(1))

        
        unique_labels, labels_count = label.unique(dim=0, return_counts=True)
        
        res = torch.zeros_like(unique_labels, dtype=torch.float, device=a.device).scatter_add_(0, label, a)
        res = res / labels_count.float().unsqueeze(1)
        prediction = self.softmax(self.linear2(res))
        
        label_y = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, y.size(1))
        
        y_ = torch.zeros([unique_labels.shape[0],y.size(1)], dtype=torch.float, device=a.device).scatter_add_(0, label_y, y)
        y_ = y_ / labels_count.float().unsqueeze(1)
        
        return self.loss(prediction, y_)
    
    def predict(self, x, labels):
        if self.use_norm != 'none':
            x = self.norm(x)
        a = torch.sigmoid((self.linear(x))) * x
        unique, labels_tmp = torch.unique(labels, sorted=True, return_inverse=True)
        label = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, a.size(1))

        
        unique_labels, labels_count = label.unique(dim=0, return_counts=True)
        
        res = torch.zeros_like(unique_labels, dtype=torch.float, device=a.device).scatter_add_(0, label, a)
        res = res / labels_count.float().unsqueeze(1)
        prediction = self.softmax(self.linear2(res))
        return prediction.detach(), unique
                         
class ContrastLinearProbing(nn.Module):
    def __init__(self, input_dim, y_dim, hidden_dim, use_norm = 'none'):
        super(ContrastLinearProbing, self).__init__()
        self.use_norm = use_norm
        if self.use_norm == 'batch':
            self.norm = nn.BatchNorm1d(input_dim)
        elif self.use_norm == 'layer':
            self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, 1)  # output the weights for each cell
        #self.linear1 = nn.Linear(hidden_dim, 1)
        self.linear2 = nn.Linear(input_dim, y_dim)  # output the weights for each cell
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels, labels2, y):
        if self.use_norm != 'none':
            x = self.norm(x)
        a = torch.sigmoid((self.linear(x))) * x

        # Get unique values of label2
        unique_label2 = torch.unique(labels2)
        
        if len(unique_label2) != 2:
            raise ValueError("label2 should have exactly two unique values")
        
        # Process for each unique value in label2
        ress = []
        uniques = []
        for val in unique_label2:
            mask = (labels2 == val)
            a_masked = a[mask]
            labels_masked = labels[mask]

            # Process based on labels within each label2 group
            unique, labels_tmp = torch.unique(labels_masked, sorted=True, return_inverse=True)
            label = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, a_masked.size(1))
            unique_labels, labels_count = label.unique(dim=0, return_counts=True)
            res = torch.zeros_like(unique_labels, dtype=torch.float, device=a_masked.device).scatter_add_(0, label, a_masked)
            res = res / labels_count.float().unsqueeze(1)
            ress.append(res)
            uniques.append(unique)
            
            label_y = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, y.size(1))
        
            y_ = torch.zeros([unique_labels.shape[0],y.size(1)], dtype=torch.float, device=a.device).scatter_add_(0, label_y, y)
            y_ = y_ / labels_count.float().unsqueeze(1)
            
        
        if not torch.all(torch.eq(uniques[0], uniques[1])):
            print(uniques[0])
            print(uniques[1])
            raise ValueError("not match")
        # Calculate the difference between the two predictions
        prediction_diff = self.softmax(self.linear2(ress[1]-ress[0]))
        
        return self.loss(prediction_diff, y_)
    
    def predict(self, x, labels, labels2):
        if self.use_norm != 'none':
            x = self.norm(x)
        a = torch.sigmoid((self.linear(x))) * x

        # Get unique values of label2
        unique_label2 = torch.unique(labels2)
        
        if len(unique_label2) != 2:
            raise ValueError("label2 should have exactly two unique values")
        
        # Process for each unique value in label2
        ress = []
        uniques = []
        for val in unique_label2:
            mask = (labels2 == val)
            a_masked = a[mask]
            labels_masked = labels[mask]

            # Process based on labels within each label2 group
            unique, labels_tmp = torch.unique(labels_masked, sorted=True, return_inverse=True)
            label = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, a_masked.size(1))
            unique_labels, labels_count = label.unique(dim=0, return_counts=True)
            res = torch.zeros_like(unique_labels, dtype=torch.float, device=a_masked.device).scatter_add_(0, label, a_masked)
            res = res / labels_count.float().unsqueeze(1)
            ress.append(res)
            uniques.append(unique)
        if not torch.all(torch.eq(uniques[0], uniques[1])):
            print(uniques[0])
            print(uniques[1])
            raise ValueError("not match")
        # Calculate the difference between the two predictions
        prediction_diff = self.softmax(self.linear2(ress[1]-ress[0]))
        
        return prediction_diff, unique                      
    
class WeightedDisentangledLinearProbing(nn.Module):
    def __init__(self, x1_dim, x2_dim, y_dim, hidden_dim, use_norm = 'none'):
        super(WeightedDisentangledLinearProbing, self).__init__()
        self.use_norm = use_norm
        if self.use_norm == 'batch':
            self.n1 = nn.BatchNorm1d(x1_dim)
            self.n2 = nn.BatchNorm1d(x2_dim)
        elif self.use_norm == 'layer':
            self.n1 = nn.LayerNorm(x1_dim)
            self.n2 = nn.LayerNorm(x2_dim)
            
        self.linear = nn.Linear(x2_dim, 1)  # output the weights for each cell
        #self.linear1 = nn.Linear(hidden_dim, 1)
        self.linear2 = nn.Linear(x1_dim, y_dim)  # output the weights for each cell
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
        self.y_dim = y_dim

    def forward(self, x1, x2, labels, y):
        if self.use_norm != 'none':
            x1 = self.n1(x1)
            x2 = self.n2(x2)
        
        a = torch.sigmoid(self.linear(x2)) * x1
        unique, labels_tmp = torch.unique(labels, sorted=True, return_inverse=True)
        label = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, a.size(1))

        
        unique_labels, labels_count = label.unique(dim=0, return_counts=True)
        
        res = torch.zeros_like(unique_labels, dtype=torch.float, device=a.device).scatter_add_(0, label, a)
        res = res / labels_count.float().unsqueeze(1)
        prediction = self.softmax(self.linear2(res))
        
        label_y = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, y.size(1))
        
        y_ = torch.zeros([unique_labels.shape[0],y.size(1)], dtype=torch.float, device=a.device).scatter_add_(0, label_y, y)
        y_ = y_ / labels_count.float().unsqueeze(1)
        
        return self.loss(prediction, y_)
    
    def predict(self, x1, x2, labels):
        if self.use_norm != 'none':
            x1 = self.n1(x1)
            x2 = self.n2(x2)
        a = torch.sigmoid(self.linear(x2)) * x1
        unique, labels_tmp = torch.unique(labels, sorted=True, return_inverse=True)
        label = labels_tmp.view(labels_tmp.size(0), 1).expand(-1, a.size(1))

        
        unique_labels, labels_count = label.unique(dim=0, return_counts=True)
        
        res = torch.zeros_like(unique_labels, dtype=torch.float, device=a.device).scatter_add_(0, label, a)
        res = res / labels_count.float().unsqueeze(1)
        prediction = self.softmax(self.linear2(res))
        return prediction.detach(), unique  
    
class LinearProbing(nn.Module):
    def __init__(self, input_dim, y_dim, use_norm = 'none'):
        super(LinearProbing, self).__init__()
        self.use_norm = use_norm
        if self.use_norm == 'batch':
            self.norm = nn.BatchNorm1d(input_dim)
        elif self.use_norm == 'layer':
            self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, y_dim)  # output the weights for each cell
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        if self.use_norm != 'none':
            x = self.norm(x)
        prediction = self.softmax(self.linear(x))
        return self.loss(prediction, y)
    
    def predict(self, x):
        if self.use_norm != 'none':
            x = self.norm(x)
        prediction = self.softmax(self.linear(x))
        return prediction.detach()