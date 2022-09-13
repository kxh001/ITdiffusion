"""
Code for Infotmation Theoretic Diffusion Model based on logisitc integrator
"""
import math
import time
import torch as t
import torch.nn as nn
from .utils import logistic_integrate
from diffusionmodel import DiffusionModel

class ITDiffusionModel(DiffusionModel):
    """Base class diffusion model for image dataset, with optional conditional info, y."""

    def __init__(self, steps, is_collect, model):
        super(ITDiffusionModel, self).__init__(model)
        self.steps = steps
        self.logs['logsnr_grid'] = t.linspace(-6, 20, 60) if not is_collect else t.tensor([-2., 2., 8., 10.])  # Where to evaluate curves for later visualization
        self.mse_grid = t.zeros(len(self.logs['logsnr_grid']))
        if is_collect:
            self.image_collection = {"x":[], "z":[], "eps":[], "eps_hat":[], "eps_diff":[]}

    def calc_nll(self, model, batch, logsnr_samples_per_x, val_len, is_viz=False, is_collect=False):
        """Given dataset, and a pre-trained model to calculate the negative log-likelihood.
           batch: [N x C x ...]
        """
        self.model = model
        # self.dataset_info(batch)
        with t.no_grad():
            # Calculate NLL
            decoder_nll = self.nll(batch, logsnr_samples_per_x)

            # high quality MSE on snr grid
            if is_viz or is_collect:
                for j, this_logsnr in enumerate(self.logs['logsnr_grid']):
                    this_logsnr_broadcast = this_logsnr * t.ones(len(batch[0]), device=batch[0].device)
                    this_mse = t.mean(self.mse(batch, this_logsnr_broadcast, mse_type='epsilon')) / val_len 
                    if is_viz:
                        self.mse_grid[j] += this_mse.cpu()

        return {"nll": decoder_nll.cpu()}

    # def collection_function(self, x, z, eps, eps_hat):
    #     """Collect interior images and estimations"""
    #     self.image_collection["x"].append(x)
    #     self.image_collection["z"].append(z)
    #     self.image_collection["eps"].append(eps)
    #     self.image_collection["eps_hat"].append(eps_hat)
    #     self.image_collection["eps_diff"].append(eps - eps_hat)