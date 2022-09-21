"""
Code for Infotmation Theoretic Diffusion Model based on logisitc integrator
"""
import math
import numpy as np
import time
import torch as t
import torch.nn as nn
from tqdm import tqdm
from .utils import logistic_integrate
from utilsiddpm import logger
from diffusionmodel import DiffusionModel

class ITDiffusionModel(DiffusionModel):
    """Base class diffusion model for image dataset, with optional conditional info, y."""

    def __init__(self, model):
        super(ITDiffusionModel, self).__init__(model)
        # self.steps = steps
        self.logs['logsnr_grid'] = t.linspace(-6, 20, 100) # if not is_collect else t.tensor([-2., 2., 8., 10.])  # Where to evaluate curves for later visualization
        self.mse_grid = t.zeros(len(self.logs['logsnr_grid']))
        self.results = {'mses':[], 'mses_round_xhat':[], 'mses_dequantize':[],
                        'mmse_g':[], 'nll (nats)':[], 'nll (nats) - dequantize':[],
                        'nll (bpd)':[], 'nll (bpd) - dequantize':[], 'nll-discrete-limit (bpd)':[],
                        'nll-discrete-limit (bpd) - dequantize':[], 'nll-discrete':[], 'nll-discrete (bpd)':[]}
        # if is_collect:
            # self.image_collection = {"x":[], "z":[], "eps":[], "eps_hat":[], "eps_diff":[]}

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

    def fit(self, dataloader_train, dataloader_test=None, epochs=10, use_optimizer='adam', lr=1e-3, verbose=False):
        """Given dataset, train the MMSE model for predicting the noise (or score).
           See CustomDataset class for example of torch dataset, that can be used with dataloader
           Shape needs to be compatible with model inputs.
        """
        if use_optimizer == 'adam':
            optimizer = t.optim.Adam(self.model.parameters(), lr=lr)
        elif use_optimizer == 'adamw':
            optimizer = t.optim.AdamW(self.model.parameters(), lr=lr)
        else:
            optimizer = t.optim.SGD(self.model.parameters(), lr=lr)

        # Return to standard fitting paradigm
        for i in tqdm(range(epochs)):  # Main training loop
            '''
            ### We can try various 'lr_scheduler' here ### 
            # lr decay -- multistep
            if i == 3:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1
            if i == 6:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1

            # lr decay -- linear
            if i % 5 == 0 and i > 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.95
            '''

            mean_loss = 0.
            t0 = time.time()
            cnt = 0
            self.train()
            for batch in dataloader_train:
                cnt += 1
                if cnt % 1000 == 0:
                    print(f"trained {cnt*len(batch[0])} samples...")
                optimizer.zero_grad()
                loss = self.nll(batch)
                loss.backward()
                optimizer.step()
                # Track running statistics
                mean_loss += loss.detach().cpu().item() / len(dataloader_train)
            self.logs['train loss'].append(mean_loss)
            iter_per_sec = len(dataloader_train) / (time.time() - t0)

            if dataloader_test:  # Process validation statistics once per epoch, if available
                self.eval()
                with t.no_grad():
                    results, val_loss = self.test_nll(dataloader_test, npoints=100, delta=1./127.5, xinterval=(-1, 1))
                    np.save(f"/home/theo/Research_Results/fine_tune/results_epoch{i}.npy", results)
                self.log_function(val_loss, results)

            if verbose:
                logger.log('epoch: {:3d}\t train loss: {:0.4f}\t val loss: {:0.4f}\t iter/sec: {:0.2f}'.
                      format(i, mean_loss, val_loss, iter_per_sec))
                logger.log('nll-discrete (bpd): {:0.4f}\t nll-discrete-limit (bpd): {:0.4f}\t nll-discrete-limit (bpd) - dequantize:{:0.4f}'.
                    format(results['nll-discrete (bpd)'], results['nll-discrete-limit (bpd)'], results['nll-discrete-limit (bpd) - dequantize']))

    def log_function(self, val_loss, results):
        """Record logs during training."""
        self.logs['logsnr_loc'].append(self.loc_logsnr)
        self.logs['logsnr_scale'].append(self.scale_logsnr)
        self.logs['val loss'].append(val_loss)
        self.logs['mse_curves'].append(results['mses'] / t.exp(self.logs['logsnr_grid']))
        self.logs['mse_eps_curves'].append(results['mses'])
        self.results['mses'] = results['mses']
        self.results['mses_round_xhat'] = results['mses_round_xhat']
        self.results['mses_dequantize'] = results['mses_dequantize']
        self.results['mmse_g'] = results['mmse_g']
        self.results['nll (nats)'] = results['nll (nats)']
        self.results['nll (nats) - dequantize'] = results['nll (nats) - dequantize']
        self.results['nll (bpd)'] = results['nll (bpd) - dequantize']
        self.results['nll-discrete-limit (bpd)'] = results['nll-discrete-limit (bpd)']
        self.results['nll-discrete-limit (bpd) - dequantize'] = results['nll-discrete-limit (bpd) - dequantize']
        self.results['nll-discrete'] = results['nll-discrete']
        self.results['nll-discrete (bpd)'] = results['nll-discrete (bpd)']

    # def collection_function(self, x, z, eps, eps_hat):
    #     """Collect interior images and estimations"""
    #     self.image_collection["x"].append(x)
    #     self.image_collection["z"].append(z)
    #     self.image_collection["eps"].append(eps)
    #     self.image_collection["eps_hat"].append(eps_hat)
    #     self.image_collection["eps_diff"].append(eps - eps_hat)