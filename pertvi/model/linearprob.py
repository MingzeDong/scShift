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
    """
        A linear probing model class supporting multiple modes of analysis.

        This class provides different linear probing strategies (Weighted,
        Contrast, WeightedDisentangled, etc.) for analyzing single-cell data.

        Parameters
        ----------
        adata : AnnData
            An AnnData object containing single-cell data. 
            The relevant embeddings/features are expected in `adata.obsm`.
        x : str
            Key in `adata.obsm` for the primary feature embedding.
        y : str
            Key in `adata.obs` for the target variable (categorical).
        label : str, optional
            Key in `adata.obs` for grouping samples or conditions.
        x2_label : str, optional
            If the mode is 'WeightedDisentangled' or 'Contrast', 
            this key indicates a secondary feature embedding or label.
        mask_label : str, optional
            If provided, is used to define distinct sets for training/validation/test,
            grouping by this label instead of individual cells.
        mode : str, default: 'Weighted'
            The linear probing mode. One of ["Weighted", "WeightedDisentangled", 
            "Contrast", "Individual"].
        norm : str, default: 'none'
            Whether to apply normalization to inputs ('none', 'batch', or 'layer').
        hidden_dim : int, default: 10
            Size of hidden dimension if needed (used in some modes).

        Returns
        -------
        None
            Initializes the probing model in the specified mode.
    """

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

        """
            Train the linear probing model.

            Splits data (or groups) into train/validation (and test, if enough leftover) 
            according to `train_size` and `validation_size`. Then fits the chosen 
            linear probing mode using an Adam optimizer.

            Parameters
            ----------
            max_epochs : int, optional
                Maximum number of training epochs. Defaults to a heuristic if not provided.
            use_gpu : Union[str, int, bool], optional
                If True or a valid GPU identifier, trains on GPU if available; else on CPU.
            train_size : float, default: 0.9
                Proportion of data (or groups) to include in the training set.
            batch_size : int, optional
                Batch size for DataLoader. If None, data is loaded in one batch.
            validation_size : float, optional
                Proportion of data (or groups) for the validation set. Defaults to `1 - train_size`.
            lr : float, default: 1e-3
                Learning rate for the optimizer.
            weight_decay : float, default: 1e-4
                Weight decay (L2 penalty).
            device : torch.device or str, optional
                Device to use. If None, automatically uses GPU if `use_gpu` is True.
            seed : int, optional
                Random seed for sampling training and validation sets. 
                If not specified, then scvi.settings.seed is used.

            Returns
            -------
            (list, list)
                A tuple of two lists: (train_loss, val_loss).
                If `validation_size > 0`, val_loss is populated; otherwise it's empty.
        """

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

        """
            Predict using the trained linear probing model.

            Depending on the `mode`, the inputs required for prediction 
            may include multiple embeddings and/or label arrays.

            Parameters
            ----------
            adata : AnnData
                The AnnData object containing embeddings/features in `adata.obsm`.
                Must have the same structure used when initializing this model.

            Returns
            -------
            Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]
                - If mode is 'Individual', returns predictions as a NumPy array.
                - Otherwise, returns (predictions, mapped_categories), where 
                  `mapped_categories` indicates the unique label groupings used.
        """

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

    """
        Internal training loop for one epoch.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model implementing a specific linear probing strategy.
        dataloader : DataLoader
            DataLoader yielding batches of (X, Y, ...) depending on mode.
        mode : str
            Probing mode, e.g. "Weighted", "Contrast", etc.
        device : str or torch.device
            Device on which computation is performed.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.

        Returns
        -------
        float
            Mean training loss for this epoch.
    """

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
    """
        Internal validation loop for one epoch.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model implementing a specific linear probing strategy.
        dataloader : DataLoader
            DataLoader yielding batches of (X, Y, ...) depending on mode.
        mode : str
            Probing mode, e.g. "Weighted", "Contrast", etc.
        device : str or torch.device
            Device on which computation is performed.

        Returns
        -------
        float
            Mean validation loss for this epoch.
    """

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
    """
        A weighted linear probing module applying per-donor averaging with learned weights.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        y_dim : int
            Dimensionality of the target (number of classes).
        hidden_dim : int
            Size of hidden dimension (unused in current minimal design, but kept for API).
        use_norm : str, default: 'none'
            Type of normalization to apply. One of ['none', 'batch', 'layer'].
    """

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

        """
            Forward pass for Weighted linear probing.

            Parameters
            ----------
            x : torch.Tensor
                Input feature of shape (batch_size, input_dim).
            labels : torch.Tensor
                Group labels for each sample, used to aggregate by group.
            y : torch.Tensor
                One-hot encoded target of shape (batch_size, y_dim).

            Returns
            -------
            torch.Tensor
                Scalar loss (cross-entropy).
        """
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

        """
            Prediction for Weighted linear probing.

            Parameters
            ----------
            x : torch.Tensor
                Input feature of shape (n_samples, input_dim).
            labels : torch.Tensor
                Group labels used for aggregation.

            Returns
            -------
            (torch.Tensor, torch.Tensor)
                - prediction: shape (n_groups, y_dim)
                - unique group labels (on the device)
        """
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
    """
        A contrastive linear probing module computing differences between two subgroups.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        y_dim : int
            Dimensionality of the target (number of classes).
        hidden_dim : int
            Hidden dimension size (unused in minimal design).
        use_norm : str, default: 'none'
            Type of normalization to apply. One of ['none', 'batch', 'layer'].

    """


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

        """
            Forward pass for contrastive linear probing.

            Parameters
            ----------
            x : torch.Tensor
                Input feature of shape (batch_size, input_dim).
            labels : torch.Tensor
                Group labels for each sample (e.g., cell type).
            labels2 : torch.Tensor
                Another label dividing data into exactly two subgroups for contrast.
            y : torch.Tensor
                One-hot encoded target of shape (batch_size, y_dim).

            Returns
            -------
            torch.Tensor
                Scalar loss.
        """


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

        """
            Prediction method for contrastive linear probing.

            Aggregates the input features by labels within each subgroup of `labels2`,
            then calculates the difference of those aggregated embeddings.

            Parameters
            ----------
            x : torch.Tensor
                Input feature of shape (n_samples, input_dim).
            labels : torch.Tensor
                Group labels for each sample.
            labels2 : torch.Tensor
                Another label dividing data into exactly two subgroups for contrast.

            Returns
            -------
            (torch.Tensor, torch.Tensor)
                - prediction_diff: The softmax of linear-probed difference.
                - unique labels used in the grouping (for alignment).
        """

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

    """
        A weighted, disentangled linear probing module that combines two embeddings.

        Parameters
        ----------
        x1_dim : int
            Dimensionality of the first input embedding.
        x2_dim : int
            Dimensionality of the second input embedding.
        y_dim : int
            Dimensionality of the target (number of classes).
        hidden_dim : int
            Hidden dimension size.
        use_norm : str, default: 'none'
            Type of normalization to apply. One of ['none', 'batch', 'layer'].
    """

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

        """
            Forward pass for WeightedDisentangled linear probing.

            Parameters
            ----------
            x1 : torch.Tensor
                First embedding input of shape (batch_size, x1_dim).
            x2 : torch.Tensor
                Second embedding input of shape (batch_size, x2_dim).
            labels : torch.Tensor
                Group labels used for aggregation.
            y : torch.Tensor
                One-hot encoded target of shape (batch_size, y_dim).

            Returns
            -------
            torch.Tensor
                Scalar cross-entropy loss.
        """
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

        """
            Prediction method for WeightedDisentangled linear probing.

            Parameters
            ----------
            x1 : torch.Tensor
                First embedding of shape (n_samples, x1_dim).
            x2 : torch.Tensor
                Second embedding of shape (n_samples, x2_dim).
            labels : torch.Tensor
                Group labels used for aggregation.

            Returns
            -------
            (torch.Tensor, torch.Tensor)
                - predictions: shape (n_groups, y_dim)
                - unique label IDs (on the device)
        """

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

    """
        A simple linear probing module without per-group weighting.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        y_dim : int
            Dimensionality of the target (number of classes).
        use_norm : str, default: 'none'
            Type of normalization to apply. One of ['none', 'batch', 'layer'].
    """

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
        """
            Forward pass for a simple linear probing.

            Parameters
            ----------
            x : torch.Tensor
                Input feature of shape (batch_size, input_dim).
            y : torch.Tensor
                One-hot encoded target of shape (batch_size, y_dim).

            Returns
            -------
            torch.Tensor
                Scalar cross-entropy loss.
        """
        if self.use_norm != 'none':
            x = self.norm(x)
        prediction = self.softmax(self.linear(x))
        return self.loss(prediction, y)
    
    def predict(self, x):

        """
            Inference for a simple linear probing.

            Parameters
            ----------
            x : torch.Tensor
                Input feature of shape (n_samples, input_dim).

            Returns
            -------
            torch.Tensor
                Predictions of shape (n_samples, y_dim).
        """

        if self.use_norm != 'none':
            x = self.norm(x)
        prediction = self.softmax(self.linear(x))
        return prediction.detach()