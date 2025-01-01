"""Model class for scShift that disentangles batch-dependent and batch-independent variations in data."""

import logging
import warnings
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import math
import torch
import scanpy as sc

import pytorch_lightning as pl
import torch.optim as optim
from scvi import settings
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from torch.distributions import Normal
from torch.autograd import Variable as V

from random import choices

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    ObsmField,
    LayerField,
    NumericalJointObsField,
    LabelsWithUnlabeledObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.dataloaders._ann_dataloader import BatchSampler
from scvi.dataloaders._anntorchdataset import AnnTorchDataset
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    scrna_raw_counts_properties,
)
from scvi.model.base import BaseModelClass, ArchesMixin
from scvi.model.base._utils import _de_core
from scvi.utils import setup_anndata_dsp
from scvi.train import SemiSupervisedTrainingPlan, TrainingPlan, TrainRunner
from scvi.dataloaders import DataSplitter, SemiSupervisedDataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._data_splitting import validate_data_split


from pertvi.module.pertvi import PertVIModule

logger = logging.getLogger(__name__)
Number = Union[int, float]


class PertVIModel(BaseModelClass, ArchesMixin):
    """
        Model class for scShift.

        This model aims to disentangle batch-dependent and batch-independent variations
        in single-cell data using variational inference.

        Parameters
        ----------
        adata : AnnData
                AnnData object containing single-cell data. Must have been set up
                via `scShift.setup_anndata` or an equivalent.
        n_batch : int, default: 0
                Number of batches in the dataset.
        n_hidden : int, default: 128
                Number of nodes per hidden layer.
        n_latent : int, default: 10
                Dimensionality of the latent space.
        n_layers : int, default: 2
                Number of hidden layers in encoder/decoder neural networks.
        dropout_rate : float, default: 0
                Dropout rate to apply to layers.
        use_observed_lib_size : bool, default: True
                If True, use observed library size as scaling factor in the mean of the distribution.
        lam_l0 : float, default: 50
                Regularization coefficient (L0) for dataset label encoding through the stochastic gate mechanism.
        lam_l1 : float, default: 0.0
                L1 penalty coefficient.
        lam_corr : float, default: 5
                Independence regularization between centralized embedding and dataset label encoding.
        var_eps : float, default: 1e-4
                Minimal variance for the variational posteriors.
        kl_weight : float, default: 1
                KL divergence scale factor.

        Returns
        -------
        None
                The model is initialized in place.

    """
    def __init__(
        self,
        adata: AnnData,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0,
        use_observed_lib_size: bool = True,
        lam_l0: float = 50,
        lam_l1: float = 0,
        lam_corr: float = 5,
        var_eps: float = 1e-4,
        kl_weight: float = 1,
    ) -> None:
        super(PertVIModel, self).__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = PertVIModule(
            n_input=self.summary_stats["n_vars"],
            n_pert = adata.obsm['pert'].shape[1],
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_output=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            lam_l0 = lam_l0,
            lam_l1 = lam_l1,
            lam_corr = lam_corr,
            var_eps = var_eps,
            kl_weight = kl_weight,
        )
        self._model_summary_string = "PertVI"
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        pert_key: str = 'pert',
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
            Set up AnnData instance for scShift model. Need to run get_pert first.

            Parameters
            ----------
            adata : AnnData
                    AnnData object containing raw counts. Rows represent cells, columns
                    represent features.
            pert_key : str, default: 'pert'
                    Key in `adata.obsm` for perturbation encoding.
            layer : str, optional
                    If not None, uses this as the key in adata.layers for raw count data.
            batch_key : str, optional
                    Key in `adata.obs` for batch information. Categories will automatically be
                    converted into integer categories.
            labels_key : str, optional
                    Key in `adata.obs` for label information.
            size_factor_key : str, optional
                    Key in `adata.obs` for size factor information. If not provided,
                    library size will be used.
            categorical_covariate_keys : List[str], optional
                    Keys in `adata.obs` corresponding to categorical data.
            continuous_covariate_keys : List[str], optional
                    Keys in `adata.obs` corresponding to continuous data.
            **kwargs
                    Additional keyword arguments for registration.

            Returns
            -------
            None
                    The `adata` is modified in place to include the necessary fields
                    for PertVIModel. An `AnnDataManager` is then registered to this class.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            ObsmField(pert_key, pert_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField(
                REGISTRY_KEYS.LABELS_KEY, labels_key, 'label_0'
            ),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        use_mask = False,
        representation_kind: str = "all",
    ) -> np.ndarray:
        """
            Return the latent representation for each cell.

            Parameters
            ----------
            adata : AnnData, optional
                    AnnData object with equivalent structure to initial AnnData. If `None`,
                    uses the AnnData object used to initialize the model.
            indices : Sequence[int], optional
                    Indices of cells in adata to use. If `None`, use all cells.
            give_mean : bool, default: True
                    Whether to return the mean of the distribution or a sampled value.
            batch_size : int, optional
                    Mini-batch size for data loading. 
            use_mask : bool, default: False
                    If True, uses a masked inference input instead of the full inference input.
            representation_kind : str, default: "all"
                    Either "base", "pert", or "all". Controls how the latent embedding is computed.

            Returns
            -------
            np.ndarray
                    A Numpy array of shape (n_cells, n_latent) containing latent representations.
        """
        available_representation_kinds = ["base", "pert","all"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )
        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices,batch_size=batch_size,shuffle=False,data_loader_class=AnnDataLoader)
        latent = []
        for tensors in dataloader:
            if use_mask:
                inference_inputs = self.module._get_inference_input(tensors)
            else:
                inference_inputs = self.module._get_inference_input_eval(tensors)
            outputs = self.module.inference(**inference_inputs)
            if representation_kind=='base':
                latent_m = outputs["q_m"]
                latent_sample = outputs["z"]
            elif representation_kind=='pert':
                latent_m = outputs["p_m"]
                latent_sample = outputs["z_pert"]
                latent_m = torch.sign(latent_m) * (torch.clamp(torch.abs(latent_m),min=0.1)-0.1)
            else:
                latent_pert = torch.sign(outputs["p_m"]) * (torch.clamp(torch.abs(outputs["p_m"]),min=0.1)-0.1)
                latent_m = outputs["q_m"] + latent_pert
                latent_sample = outputs["z_all"]

            if give_mean:
                latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
            
        return torch.cat(latent).numpy()


    
        
    def get_pert(
        adata,
        pert_label = None,
        drug_label = None,
        dose_label = None,
        ct_pert = None,
        ct_drug = None,
    ):
        """
            A simple function to create one-hot dataset label encoding and store in `adata.obsm['pert']`. 

            Parameters
            ----------
            adata : AnnData
                    The AnnData object.
            pert_label : str, optional
                    Key in `adata.obs` corresponding to drug / data identity.
            drug_label : str, optional
                    Key in `adata.obs` corresponding to drug / data identity.
            dose_label : str, optional
                    Key in `adata.obs` for the dosage levels.
            ct_pert : str, optional
                    Name or category representing control (unperturbed) in `pert_label` (Depracated).
            ct_drug : str, optional
                    Name or category representing control drug in `drug_label` (Depracated).

            Returns
            -------
            None
                    Modifies `adata.obsm['pert']` in place with the new encoding.
        """
        if pert_label is None:
            df = pd.get_dummies(adata.obs[drug_label]) * 1
            if dose_label is not None:
                df = df * adata.obs[dose_label][:,None]
            #df.iloc[adata.obs[drug_label]==ct_drug] = 0
        elif drug_label is None:
            df = pd.get_dummies(adata.obs[pert_label]) * 1
            #df.iloc[adata.obs[pert_label]==ct_pert] = 0  
        adata.obsm['pert'] = df.values
        return

    
    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        n_samples_per_label = 100,
        lr = 1e-3,
        weight_decay = 1e-4,
        n_epochs_kl_warmup = None,
        n_steps_kl_warmup = 1600,
        **trainer_kwargs,
    ) -> None:
        """
            Train the scShift model using a semi-supervised data splitter.

            This method sets up a training loop with the chosen data splitter and
            training plan. It can optionally perform early stopping if desired.

            Parameters
            ----------
            max_epochs : int, optional
                    Number of passes through the dataset. Defaults to a heuristic based on
                    the number of cells if not specified.
            use_gpu : Union[str, int, bool], optional
                    Whether to use GPU for training. Can be None, True/False, or a specific
                    GPU index or name, e.g. "cuda:0".
            batch_size : int, default: 128
                    Mini-batch size for data loading during training.
            early_stopping : bool, default: False
                    If True, perform early stopping based on validation loss.
            train_size : float, default: 0.9
                    Proportion of cells to include in the training set.
            validation_size : float, optional
                    Proportion of cells to include in the validation set. If None, uses
                    1 - train_size. Additional cells, if any, form a test set.
            n_samples_per_label : int, default: 100
                    Number of labeled samples to use per label category in semi-supervised mode.
            lr : float, default: 1e-3
                    Learning rate for the optimizer.
            weight_decay : float, default: 1e-4
                    Weight decay for the optimizer, acting as an L2 regularization.
            n_epochs_kl_warmup : int, optional
                    Number of epochs over which to scale up the KL term from 0 to 1.
            n_steps_kl_warmup : int, default: 1600
                    Number of training steps over which to warm up the KL divergence term.
            **trainer_kwargs
                    Additional keyword arguments passed to the :class:`~scvi.train.Trainer`
                    or :class:`~scvi.train.SemiSupervisedTrainingPlan`.

            Returns
            -------
            None
                    The model is trained in place. Check logs for training progress or
                    potential early stopping triggers.
        """
        data_splitter = SemiSupervisedDataSplitter(
            self.adata_manager,
            n_samples_per_label=n_samples_per_label,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        #data_splitter = DataSplitter(
        #    self.adata_manager,
        #    train_size=train_size,
        #    validation_size=validation_size,
        #    batch_size=batch_size,
        #    use_gpu=use_gpu,
        #)
        
        #training_plan = TrainingPlan(self.module, lr = lr, weight_decay = weight_decay,n_steps_kl_warmup = n_steps_kl_warmup, n_epochs_kl_warmup = n_epochs_kl_warmup)
        training_plan = SemiSupervisedTrainingPlan(self.module, lr = lr, weight_decay = weight_decay,n_steps_kl_warmup = n_steps_kl_warmup, n_epochs_kl_warmup = n_epochs_kl_warmup)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
    
    
    
    

        
        
        