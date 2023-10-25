"""Model class for pertVI for single cell expression data."""

import logging
import warnings
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import math
import torch
import scanpy as sc
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import Delaunay
import pytorch_lightning as pl
import torch.optim as optim
from scvi import settings
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from torch.distributions import Normal
from torch.autograd import Variable as V

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
from scvi.train import SemiSupervisedTrainingPlan, TrainRunner
from scvi.dataloaders import DataSplitter, SemiSupervisedDataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._data_splitting import validate_data_split


from pertvi.module.pertvi import PertVIModule

logger = logging.getLogger(__name__)
Number = Union[int, float]


class PertVIModel(BaseModelClass, ArchesMixin):
    """
    Model class for pertVI.
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
        lam_l1: float = 1e-2,
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
        Set up AnnData instance for MetaVI model.

        Args:
        ----
            adata: AnnData object containing raw counts. Rows represent cells, columns
                represent features.
            layer: If not None, uses this as the key in adata.layers for raw count data.
            batch_key: Key in `adata.obs` for batch information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_batch"]`. If None, assign the same batch to all the
                data.
            labels_key: Key in `adata.obs` for label information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_labels"]`. If None, assign the same label to all the
                data.
            size_factor_key: Key in `adata.obs` for size factor information. Instead of
                using library size as a size factor, the provided size factor column
                will be used as offset in the mean of the likelihood. Assumed to be on
                linear scale.
            categorical_covariate_keys: Keys in `adata.obs` corresponding to categorical
                data. Used in some models.
            continuous_covariate_keys: Keys in `adata.obs` corresponding to continuous
                data. Used in some models.

        Returns
        -------
            If `copy` is True, return the modified `adata` set up for MetaVI
            model, otherwise `adata` is modified in place.
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

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to full batch training.
        representation_kind: "intrinsic", "interaction" or "all" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
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

    @torch.no_grad()
    def get_response(
        self,
        adata: Optional[AnnData] = None,
        pert_key: str = 'pert',
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        use_mask = False,
        batch_size: Optional[int] = None,
        representation_kind: str = "all",
    ) -> np.ndarray:
        """
        After training, return predicted response at gene level. 
        """
        adata_ = adata.copy()
        adata_.obsm['pert'] = adata.obsm[pert_key].copy()
        adata_ = self._validate_anndata(adata_)
        dataloader = self._make_data_loader(adata=adata_, indices=indices,batch_size=batch_size,shuffle=False,data_loader_class=AnnDataLoader)
        expression = []
        for tensors in dataloader:
            if use_mask:
                inference_inputs = self.module._get_inference_input(tensors)
            else:
                inference_inputs = self.module._get_inference_input_eval(tensors)
            outputs = self.module.inference(**inference_inputs)
            g_inputs = self.module._get_generative_input(tensors, outputs)
            g_outputs = self.module.generative(**g_inputs)

            expression += [g_outputs["px_rate"].detach().cpu()]
            
        return torch.cat(expression).numpy()
    
    
    @torch.no_grad()
    def get_shift(
        self,
        adata: Optional[AnnData] = None,
        pert_key: str = 'pert',
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        use_mask = False,
        batch_size: Optional[int] = None,
        representation_kind: str = "all",
    ) -> np.ndarray:
        """
        A multi-step function that returns a perturbation list for explaining the difference between diseased states and normal states. First use CINEMA-OT to get the counterfactual pairs. Then use the trained model, together with scArches fine tuning to identify the perturbation factor.
        """
        adata_ = adata.copy()
        adata_.obsm['pert'] = adata.obsm[pert_key].copy()
        adata_ = self._validate_anndata(adata_)
        dataloader = self._make_data_loader(adata=adata_, indices=indices,batch_size=batch_size,shuffle=False,data_loader_class=AnnDataLoader)
        expression = []
        for tensors in dataloader:
            if use_mask:
                inference_inputs = self.module._get_inference_input(tensors)
            else:
                inference_inputs = self.module._get_inference_input_eval(tensors)
            outputs = self.module.inference(**inference_inputs)
            g_inputs = self.module._get_generative_input(tensors, outputs)
            g_outputs = self.module.generative(**g_inputs)

            expression += [g_outputs["px_scale"].detach().cpu()]
            
        return torch.cat(expression).numpy()    
    
    
    
    def get_pert(
        adata,
        pert_label = None,
        drug_label = None,
        dose_label = None,
        ct_pert = None,
        ct_drug = None,
    ):
        if pert_label is None:
            df = pd.get_dummies(adata.obs[drug_label]) * 1
            if dose_label is not None:
                df = df * adata.obs[dose_label][:,None]
            #df.iloc[adata.obs[drug_label]==ct_drug] = 0
        elif drug_label is None:
            df = pd.get_dummies(adata.obs[pert_label]) * 1
            #df.iloc[adata.obs[pert_label]==ct_pert] = 0
        else:
            adata.obs['pert_drug'] = adata.obs[pert_label].astype(str) + adata.obs[drug_label].astype(str)
            df = pd.get_dummies(adata.obs['pert_drug']) * 1
            if dose_label is not None:
                df = df * adata.obs[dose_label][:,None]
            #df.iloc[adata.obs[drug_label]==ct_drug] = 0
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
        **trainer_kwargs,
    ) -> None:
        """
        Train the MetaVI model. In our setting, we consider full-batch training, therefore
        we rewrite the training function. 

        Args:
        ----
        
            max_epochs: Number of passes through the dataset. If `None`, default to
                `np.min([round((20000 / n_cells) * 400), 400])`.
            use_gpu: Use default GPU if available (if `None` or `True`), or index of
                GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
                or use CPU (if `False`).
            train_size: Size of training set in the range [0.0, 1.0].
            validation_size: Size of the validation set. If `None`, default to
                `1 - train_size`. If `train_size + validation_size < 1`, the remaining
                cells belong to the test set.
            batch_size: Mini-batch size to use during training.
            early_stopping: Perform early stopping. Additional arguments can be passed
                in `**kwargs`. See :class:`~scvi.train.Trainer` for further options.
            plan_kwargs: Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword
                arguments passed to `train()` will overwrite values present
                in `plan_kwargs`, when appropriate.
            **trainer_kwargs: Other keyword args for :class:`~scvi.train.Trainer`.

        Returns
        -------
            None. The model is trained.
        """
        data_splitter = SemiSupervisedDataSplitter(
            self.adata_manager,
            n_samples_per_label=n_samples_per_label,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = SemiSupervisedTrainingPlan(self.module, lr = lr, weight_decay = weight_decay)

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

    def query_base(
        self,
        adata_query,
        model,
        indices: Optional[Sequence[int]] = None,
        batch_size: int = 128,
        use_mask = False,
    ) -> None:
        adata_prepared = adata_query.copy()
        adata_prepared.obs_names_make_unique()
        adata_prepared.var_names_make_unique()
        self.prepare_query_anndata(adata_prepared,model)
        
        adata_prepared.obsm['pert'] = np.zeros((adata_prepared.shape[0],self.module.n_pert))
        
        return self.get_latent_representation(adata_prepared,representation_kind='base')
    
    
    
    def query_pert(
        self,
        adata_query,
        model,
        max_epochs: Optional[int] = 50,
        use_gpu: Optional[Union[str, int, bool]] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: int = 128,
        use_mask = False,
        early_stopping: bool = False,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        n_samples_per_label = 100,
        lr = 1e-3,
        weight_decay = 1e-4,
        **trainer_kwargs,
    ) -> None:
        ### Here we learn the optimal z_pert in an zero-shot setting.
        ### Then the original latent space is regressed to achieve independence with z_pert.
        ### Here the setting is different from scArches and sVAE.
        
        
        
        adata_prepared = adata_query.copy()
        adata_prepared.obs_names_make_unique()
        adata_prepared.var_names_make_unique()
        self.prepare_query_anndata(adata_prepared,model)
        
        adata_prepared.obsm['pert'] = np.zeros((adata_prepared.shape[0],self.module.n_pert))
        
        adata_prepared = self._validate_anndata(adata_prepared)
        gpus, device = parse_use_gpu_arg(use_gpu, return_device=True)
        pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and gpus != 0) else False
        )
        
        tensors = AnnTorchDataset(self.get_anndata_manager(adata_prepared, required=True))
        #dataloader = self._make_data_loader(adata=adata_prepared, indices=indices,batch_size=adata_prepared.shape[0],shuffle=False,pin_memory=pin_memory,data_loader_class=AnnDataLoader)
        #for tensors in dataloader:
        #    for ind in tensors:
        #        tensors[ind] = tensors[ind].to(model.device)
        
        sampler_kwargs = {
            "indices": np.arange(adata_prepared.shape[0]),
            "batch_size": batch_size,
            "shuffle": True,
            "drop_last": False,
        }
        
        sampler = BatchSampler(**sampler_kwargs)
        ## initialization
        
        p_m_ = torch.rand(adata_prepared.shape[0],self.module.n_output,requires_grad=True, device=model.device)
        p_v_ = torch.rand(adata_prepared.shape[0],self.module.n_output,requires_grad=True, device=model.device)
        #p_m = p_m_ - 0.5
        
        self.module.freeze_params()
        
        optimizer = optim.Adam([p_m_,p_v_], lr=lr, weight_decay=weight_decay)
        
        pbar = tqdm(range(1, max_epochs + 1))
        
        train_loss = []
        
        for epoch in pbar:
            train_loss.append(train_query(self.module, tensors, sampler, p_m_, p_v_, optimizer, use_mask).detach())
            pbar.set_description('Epoch '+str(epoch)+'/'+str(max_epochs))
            pbar.set_postfix(train_loss=train_loss[epoch-1].cpu().numpy())
            #print('Epoch ',epoch)
            
                
        model.module.eval()
        ## Now return trained p_m as the pert_embedding for the model
        p_m = torch.sign(p_m_-0.5) * (torch.clamp(torch.abs(p_m_-0.5),min=0.1)-0.1)
        p_m = p_m.detach().cpu().numpy()
        #init_p_m = p_m - p_m.mean(axis=0)
        #init_p_m = init_p_m / init_p_m.std(axis=0)

        base_rep = self.get_latent_representation(adata_prepared,representation_kind='base')
        #pert_embedding = self.get_latent_representation(self.adata_manager.adata,representation_kind='pert')
        #id_ind = np.abs(pert_embedding).sum(axis=0).copy()
        #base_rep[:,id_ind>0] = base_rep[:,id_ind>0] - base_rep[:,id_ind>0].mean(axis=0)
        #base_rep[:,id_ind>0] = base_rep[:,id_ind>0] / base_rep[:,id_ind>0].std(axis=0)
        #base_rep[:,id_ind>0] = base_rep[:,id_ind>0] - init_p_m[:,id_ind>0] * (init_p_m[:,id_ind>0] * base_rep[:,id_ind>0]).sum(axis=0) / init_p_m.shape[0]
        
        #base_rep[:,id_ind==0] = base_rep[:,id_ind==0] + p_m[:,id_ind==0]
        #p_m[:,id_ind==0] = 0
        
        return base_rep + p_m
    
    
def train_query(model, tensors, sampler, p_m_, p_v_, optimizer, use_mask):
    model.train()
    for idx in sampler:
        tensor_tmp = tensors[idx]
        for ind in tensor_tmp:
            tensor_tmp[ind] = torch.Tensor(tensor_tmp[ind]).to(model.device)
        
        optimizer.zero_grad()
        if use_mask:
            inference_inputs = model._get_inference_input(tensor_tmp)
        else:
            inference_inputs = model._get_inference_input_eval(tensor_tmp)
        outputs = model.inference(**inference_inputs)
        p_m = torch.sign(p_m_-0.5) * (torch.clamp(torch.abs(p_m_-0.5),min=0.1)-0.1)
        p_v = p_v_ + 0.01
        outputs['p_m'] = p_m[idx]
        outputs['p_v'] = p_v[idx]
        dist = Normal(p_m[idx], p_v[idx].sqrt())
        z_pert = dist.rsample()
        outputs['z_pert'] = z_pert
        outputs['z_all'] = z_pert + outputs['z']
    
        g_inputs = model._get_generative_input(tensor_tmp, outputs)
        g_outputs = model.generative(**g_inputs)
        lossrecorder = model.loss(tensor_tmp, outputs, g_outputs)
        loss = lossrecorder.loss
    
        loss.backward(retain_graph=True)
        optimizer.step()
    return loss

        
        
        