"""PyTorch module for scShift (PertVI) that disentangles batch-dependent and batch-independent variations in data."""

from typing import Dict, Optional, Tuple

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.autograd import Variable

import torch.optim as optim

torch.backends.cudnn.benchmark = True


class PertVIModule(BaseModuleClass):

    def __init__(
        self,
        n_input: int,
        n_pert: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_output: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        kl_weight: float = 1,
        lam_l0: float = 50,
        lam_corr: float = 5,
        lam_l1: float = 0.01,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_pert = n_pert
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene-batch"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.use_observed_lib_size = use_observed_lib_size
        self.var_eps = var_eps
        self.lam_l0 = lam_l0
        self.lam_l1 = lam_l1
        self.lam_corr = lam_corr
        self.kl_weight_ = kl_weight

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        cat_list = [n_batch]

        self.base_encoder = Encoder(
            n_input,
            n_output,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=False,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
            var_eps = var_eps
        )
        
        self.pert_encoder = Encoder(
            n_pert,
            n_output,
            n_cat_list=None,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            use_batch_norm=False,
            bias=False,
            use_layer_norm=False,
            var_activation=None,
            var_eps = 0.01
        )

        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )

        self.decoder = DecoderSCVI(
            n_output,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))
    
    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        # let us fetch the raw counts, and add them to the dictionary
        x = tensors[REGISTRY_KEYS.X_KEY]
        x_mask = x.clone()
        mask = torch.rand(x.shape[1]) < 0.25

        x_mask[:,mask] = x[torch.argsort(torch.rand(x.shape[0],mask.sum()),axis=0),mask].clone()
        #x_mask[:,mask] = x_mask[:,mask] * 0
        
        #x_mask[x<0] = x[x<0].clone()
        #x_mask[:,mask] = x_mask[torch.randperm(x_mask.shape[0])[:,None],mask]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        pert = tensors['pert']
        obs_libsize = torch.log(x.sum(1)).unsqueeze(1)
        input_dict = dict(x=x_mask,batch_index=batch_index,pert=pert,obs_libsize=obs_libsize)
        return input_dict
    
    def _get_inference_input_eval(self, tensors):
        """Parse the dictionary to get appropriate args"""
        # let us fetch the raw counts, and add them to the dictionary
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        pert = tensors['pert']
        obs_libsize = torch.log(x.sum(1)).unsqueeze(1)
        input_dict = dict(x=x,batch_index=batch_index,pert=pert,obs_libsize=obs_libsize)
        return input_dict
    
    def _get_generative_input(self, tensors, inference_outputs):
        z_all = inference_outputs["z_all"]
        library = inference_outputs["library"]
        batch_index = inference_outputs["batch_index"]
        input_dict = {
            "z_all": z_all,
            "library": library,
            "batch_index": batch_index,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        pert: torch.Tensor,
        obs_libsize: torch.Tensor,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        x_ = x
        if self.use_observed_lib_size:
            library = obs_libsize
        x_ = torch.log(1 + x_)

        q_m, q_v, z = self.base_encoder(x_,batch_index)
        
        pert_ = torch.log(1 + pert)
        
        p_m, p_v, z_pert = self.pert_encoder(pert_)

        z_pert = (torch.clamp(torch.sign(p_m) * z_pert, min=0.1)-0.1) * torch.sign(p_m)
        
        z_all = z + z_pert

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(x_, batch_index)
            library = library_encoded


        outputs = dict(
            z_all=z_all,
            z=z,
            q_m=q_m,
            q_v=q_v,
            z_pert=z_pert,
            pert=pert,
            p_m=p_m,
            p_v=p_v,
            library=library,
            ql_m=ql_m,
            ql_v=ql_v,
            batch_index = batch_index,
        )
        return outputs
    
    def freeze_params(self):
        # freeze
        for param in self.parameters():
            param.requires_grad = False
        
        for _, mod in self.decoder.named_modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.momentum = 0
        
        for _, mod in self.base_encoder.named_modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.momentum = 0
        
        for _, mod in self.l_encoder.named_modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.momentum = 0        


    @auto_move_data
    def generative(
        self,
        z_all: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            z_all,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )
    

    @staticmethod
    def reconstruction_loss(
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute likelihood loss for negative binomial distribution. 

        Args:
        ----
            x: Input data.
            px_rate: Mean of distribution.
            px_r: Inverse dispersion.
            px_dropout: Logits scale of zero inflation probability.

        Returns
        -------
            Negative log likelihood (reconstruction loss) for each data point. If number
            of latent samples == 1, the tensor has shape `(batch_size, )`. If number
            of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss
    
    @staticmethod
    def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and prior Gaussian.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
            prior_mean: Mean of the prior Gaussian.
            prior_var: Variance of the prior Gaussian.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    @auto_move_data
    def _compute_local_library_params(
        self, batch_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    def library_kl_divergence(
        self,
        batch_index: torch.Tensor,
        variational_library_mean: torch.Tensor,
        variational_library_var: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    @staticmethod
    def l0_loss(
        p_m: torch.Tensor,
        p_v: torch.Tensor,
    ) -> torch.Tensor:
        l = (1 + torch.erf(((torch.abs(p_m)-0.1) / p_v) / math.sqrt(2))) / 2
        #l = (1 + torch.erf((torch.abs(q_m) / q_v) / math.sqrt(2))) / 2
        values, indices = torch.kthvalue(-l,1,dim=-1)
        #l_ = (1 + torch.erf(((torch.abs(q_m)-0.1).sum(dim=0) / q_v.sum(dim=0)) * 5)) / 2
        return l.sum(dim=-1) + 2 * values
    
    @staticmethod
    def l0_loss_(
        q_m: torch.Tensor,
        q_v: torch.Tensor,
    ) -> torch.Tensor:
        l = (1 + torch.erf((torch.abs(q_m) / q_v) / math.sqrt(2))) / 2
        return l.sum(dim=-1)    
    

    def _generic_loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x = torch.Tensor(tensors[REGISTRY_KEYS.X_KEY])
        batch_index = torch.Tensor(tensors[REGISTRY_KEYS.BATCH_KEY])

        q_m = inference_outputs["q_m"]
        q_v = inference_outputs["q_v"]
        p_m = inference_outputs["p_m"]
        p_v = inference_outputs["p_v"]
        pert = inference_outputs["pert"]
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        prior_z_m = torch.zeros_like(q_m)
        prior_z_v = 0.25 * torch.ones_like(q_v)
        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)
        l0_loss = self.l0_loss(p_m,p_v)
        p_ = p_m.detach()
        p_ = (torch.clamp(torch.sign(p_) * p_, min=0.1)-0.1) * torch.sign(p_)

        
        corr_loss = self.corr_loss(z,p_)
        kl_z = self.latent_kl_divergence(q_m, q_v, prior_z_m, prior_z_v)
        kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        return dict(
            recon_loss=recon_loss,
            kl_z=kl_z,
            l0_loss=l0_loss,
            corr_loss=corr_loss,
            kl_library=kl_library,
        )

    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
        kl_weight: float = 1.0,
        **loss_kwargs,
    ) -> LossRecorder:

        losses = self._generic_loss(
            tensors,
            inference_outputs,
            generative_outputs,
        )
        reconst_loss = losses["recon_loss"]
        l0_loss = losses["l0_loss"]
        corr_loss = losses["corr_loss"]
        kl_divergence = losses["kl_z"]
        kl_divergence_l = losses["kl_library"]


        weighted_kl_local = self.lam_l0 * l0_loss + kl_divergence_l + self.kl_weight_ * kl_divergence + self.lam_corr * corr_loss

        loss = torch.mean(reconst_loss + kl_weight * weighted_kl_local)
        
        decoder_params = torch.cat([x.view(-1) for x in self.decoder.parameters()])
        
        loss = loss + self.lam_l1 * torch.norm(decoder_params, 1)

        kl_local = dict(
            kl_divergence = kl_divergence,
            kl_divergence_l = kl_divergence_l
        )
        kl_global = corr_loss

        # LossRecorder internally sums the `reconst_loss`, `kl_local`, and `kl_global`
        # terms before logging, so we do the same for our `wasserstein_loss` term.
        return LossRecorder(
            loss,
            reconst_loss,
            kl_local,
            kl_global,
        )

    def corr_loss(
        self,
        z: torch.Tensor,
        p_: torch.Tensor,
    ):
        true_samples1 = Variable(p_ + 0.5 * torch.randn(z.shape[0], z.shape[1],device=z.device),requires_grad=False)
        true_samples2 = Variable(0.5 * torch.randn(z.shape[0], z.shape[1],device=z.device),requires_grad=False)
        l1 = self.compute_mmd(z+p_,true_samples1)
        l2 = self.compute_mmd(z,true_samples2)

        return l1 + l2
    
    
    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd
    
    @staticmethod
    def compute_kernel(x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    
    