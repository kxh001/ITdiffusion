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
        self.logs['logsnr_grid'] = t.linspace(-6, 20, 100)
        self.mse_grid = t.zeros(len(self.logs['logsnr_grid']))
        self.results = {'mses':[], 'mses_round_xhat':[], 'mses_dequantize':[],
                        'mmse_g':[], 'nll (nats)':[], 'nll (nats) - dequantize':[],
                        'nll (bpd)':[], 'nll (bpd) - dequantize':[], 'nll-discrete-limit (bpd)':[],
                        'nll-discrete-limit (bpd) - dequantize':[], 'nll-discrete':[], 'nll-discrete (bpd)':[]}

    def mse_test(self, batch, logsnr, mse_type='epsilon'):
        x = batch[0].to(self.device)
        z, eps = self.noisy_channel(x, logsnr)
        noisy_batch = [z, ] + [batch[1:]]  # batch may include conditioning on y
        tList = t.linspace(0, 1000, 4000)
        mseList = []
        for this_t in tList:
            # if using iddpm model
            model_output = self.model(noisy_batch[0], this_t.view(1,).to(self.device))
            C = model_output.shape[1] // 2
            eps_hat = t.split(model_output, C, dim=1)[0]

            # if using huggingface model
            # eps_hat = self.model(noisy_batch[0], this_t)['sample'] 

            err = (eps - eps_hat).flatten(start_dim=1)  # Flatten for, e.g., image data
            mse_epsilon = t.einsum('ij,ij->i', err, err)  # MSE for epsilon
            if mse_type == 'epsilon':
                mseList.append(mse_epsilon)  # integrating over logsnr cancels out snr, so we can use mse_epsilon directly
            elif mse_type == 'x':
                mseList.append(mse_epsilon / t.exp(logsnr))  # Special form of x_hat leads to this simplification
        return mseList

    def fit_old(self, dataset, val_dataset=None, epochs=10, batch_size=200, use_optimizer='adam', lr=1e-3, verbose=False):
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

        # ONLY NON-STANDARD STEP: Essentially we have to first "fit" the base model (a Gaussian), and store data stats
        # For very high-d datasets, we will have to do this differently (Gaussian covariance is too big)
        dataloader = t.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        self.dataset_info(dataloader)  # Calculate/record useful stats about data - dimension/shape/eigenvalues

        # Return to standard fitting paradigm
        dataloader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if val_dataset:
            val_dataloader = t.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        for i in range(epochs):  # Main training loop
            mean_loss = 0.
            ### We can try various 'lr_scheduler' here ### 
            # lr decay -- multistep
            if i == 100:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1
            if i == 150:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1

            # lr decay -- linear
            # if i % 5 == 0 and i > 0:
            #     for p in optimizer.param_groups:
            #         p['lr'] *= 0.95
            t0 = time.time()
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self.nll(batch)
                loss.backward()
                optimizer.step()

                # Track running statistics
                mean_loss += loss.detach().cpu().item() / len(dataloader)
                self.logs['train loss'].append(loss.detach().cpu())
            iter_per_sec = len(dataloader) / (time.time() - t0)

            if val_dataset:  # Process validation statistics once per epoch, if available
                self.eval()
                with t.no_grad():
                    val_loss = 0.  # val. loss estimated same as train
                    mse_grid = t.zeros(len(self.logs['logsnr_grid']))  # And MSEs on SNR grid, to see curves
                    for batch in val_dataloader:
                        val_loss += self.nll(batch).cpu() / len(val_dataloader)

                        # high quality MSE on snr grid
                        for j, this_logsnr in enumerate(self.logs['logsnr_grid']):
                            this_logsnr_broadcast = this_logsnr * t.ones(len(batch[0]), device=batch[0].device)
                            this_mse = t.mean(self.mse(batch, this_logsnr_broadcast, mse_type='epsilon')) / len(val_dataloader)
                            mse_grid[j] += this_mse.cpu()

                self.log_function(val_loss, mse_grid)
                self.train()
            if verbose:
                print('epoch: {:3d}\t train loss: {:0.4f}\t val loss: {:0.4f}\t iter/sec: {:0.2f}'.
                      format(i, mean_loss, val_loss, iter_per_sec))
