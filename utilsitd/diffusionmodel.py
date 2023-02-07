"""
Code for Diffusion model class
"""
import os
import math
import time
import numpy as np
import torch as t
import torch.nn as nn
from tqdm import tqdm

from .utils import logistic_integrate, soft_round
from . import logger

class DiffusionModel(nn.Module):
    """Base class diffusion model for x, with optional conditional info, y.
       *logsnr integration: we do integrals in terms of logsnr instead of snr.
       Therefore we have to include a factor of 1/snr in integrands."""

    def __init__(self, model):
        super().__init__()
        self.model = model  # A model that takes in model(x, y (optional), snr, is_simple) and outputs the noise estimate
        self.logs = {"val loss": [], "train loss": []} # store stuff for plotting, could use tensorboard for larger models
        self.loc_logsnr, self.scale_logsnr = 0., 2.  # initial location and scale for integration. Reset in data-driven way by dataset_info
        self.clip = 4 # initial quantile for integration.
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.dtype, self.d, self.shape, self.left = None, None, None, None # dtype, dimensionality, and shape for data, set when we see it in "fit"

    def forward(self, batch, logsnr):
        """Batch is either [x,y] or [x,] depending on whether it is a conditional model."""
        return self.model(batch, t.exp(logsnr))

    def noisy_channel(self, x, logsnr):
        logsnr = logsnr.view(self.left)  # View for left multiplying
        eps = t.randn((len(logsnr),) + self.shape, dtype=x.dtype, device=x.device)
        return t.sqrt(t.sigmoid(logsnr)) * x + t.sqrt(t.sigmoid(-logsnr)) * eps, eps

    def mse(self, batch, logsnr, mse_type='epsilon', xinterval=None, delta=None, soft=False):
        """Return MSE curve either conditioned or not on y.
        x_hat = z/sqrt(snr) - eps_hat(z, snr)/sqrt(snr),
        so x_hat - x = (eps - eps_hat(z, snr))/sqrt(snr).
        And we actually reparametrize eps_hat to depend on eps(z/sqrt(1+snr), snr)
        Options are:
        mse_type: {"epsilon" or "x"), for error in predicting noise, or predicting data.
        xinterval: Whether predictions should be clamped to some interval.
        delta: If provided, round to the nearest discrete value
        """
        x = batch[0].to(self.device)  # assume iterator gives other things besides x in list (e.g. y)
        z, eps = self.noisy_channel(x, logsnr)
        noisy_batch = [z, ] + batch[1:]  # batch may include conditioning on y
        eps_hat = self.model(noisy_batch, t.exp(logsnr))
        x_hat = t.sqrt(1 + t.exp(-logsnr.view(self.left))) * z - eps_hat * t.exp(-logsnr.view(self.left) / 2)
        if delta:
            if soft:
                x_hat = soft_round(x_hat, t.exp(logsnr).view(self.left), xinterval, delta) # soft round to nearest discrete value
            else:
                x_hat = delta * t.round((x_hat - xinterval[0]) / delta) + xinterval[0]  # hard round to nearest discrete value
        if xinterval:
            x_hat = t.clamp(x_hat, xinterval[0], xinterval[1])  # clamp predictions to not fall outside of range
        err = (x - x_hat).flatten(start_dim=1)  # Flatten for, e.g., image data
        mse_x = t.einsum('ij,ij->i', err, err)  # MSE for epsilon
        if mse_type == 'epsilon':
            return mse_x * t.exp(logsnr)  # Special form of x_hat, eps_hat leads to this relation
        elif mse_type == 'x':
            return mse_x

    def nll(self, batch, logsnr_samples_per_x=1, xinterval=None):
        """-log p(x) (or -log p(x|y) if y is provided) estimated for a batch."""
        nll = 0
        for _ in range(logsnr_samples_per_x):
            logsnr, w = logistic_integrate(len(batch[0]), *self.loc_scale, device=self.device)
            mses = self.mse(batch, logsnr, mse_type='epsilon', xinterval=xinterval)
            nll += self.loss(mses, logsnr, w) / logsnr_samples_per_x
        return nll

    def loss(self, mses, logsnr, w):
        """
        Returns the (per-sample) losses from MSEs, for convenience adding constants to match
        with NLL expression.
        :param mses:  Mean square error for *epsilon*
        :param logsnr:  Log signal to noise ratio
        :param w:  Integration weights
        :return: loss, -log p(x) estimate
        """
        mmse_gap = mses - self.mmse_g(logsnr)  # The "scale" does not change the integral, but may be more numerically stable.
        loss = self.h_g + 0.5 * (w * mmse_gap).mean()
        return loss  # *logsnr integration, see paper

    @t.no_grad()
    def test_nll(self, dataloader, npoints=100, delta=None, xinterval=None, soft=False):
        """Calculate expected NLL on data at test time.  Main difference is that we can clamp the integration
        range, because negative values of MMSE gap we can switch to Gaussian decoder to get zero.
        npoints - number of points to use in integration
        delta - if the data is discrete, delta is the gap between discrete values.
        E.g. delta = 1/127.5 for CIFAR data (0, 255) scaled to -1, 1 range
        range = a tuple of the range of the discrete values, e.g. (-1, 1) for CIFAR10 normalized
        """
        if self.model.training:
            print("Warning - estimating test NLL but model is in train mode")
        results = {}  # Return multiple forms of results in a dictionary
        clip = self.clip 
        loc, scale = self.loc_scale
        logsnr, w = logistic_integrate(npoints, loc=loc, scale=scale, clip=clip, device=self.device, deterministic=True)
        left_logsnr, right_logsnr = loc - clip * scale, loc + clip * scale
        # sort logsnrs along with weights
        logsnr, idx = logsnr.sort()
        w = w[idx].to('cpu')

        ## select 100 logsnrs out of 1k (see B.5 VARIANCE ESTIMATES)
        # t.manual_seed(1024)
        # idx_r = t.randint(0, npoints, (100,))
        # logsnr = logsnr[idx_r]
        # w = w[idx_r]


        results['logsnr'] = logsnr.to('cpu')
        results['w'] = w
        mses, mses_dequantize, mses_round_xhat = [], [], []  # Store all MSEs, per sample, logsnr, in an array
        total_samples = 0
        val_loss = 0
        for batch in tqdm(dataloader):
            data = batch[0].to(self.device)  # assume iterator gives other things besides x in list
            if delta:
                data_dequantize = data + delta * (t.rand_like(data) - 0.5)
            n_samples = len(data)
            total_samples += n_samples

            val_loss += self.nll([data, ] + batch[1:], xinterval=xinterval).cpu() * n_samples

            mses.append(t.zeros(n_samples, len(logsnr)))
            mses_round_xhat.append(t.zeros(n_samples, len(logsnr)))
            mses_dequantize.append(t.zeros(n_samples, len(logsnr)))
            for j, this_logsnr in enumerate(logsnr):
                this_logsnr_broadcast = this_logsnr * t.ones(len(data), device=self.device)

                # Regular MSE, clamps predictions, but does not discretize
                this_mse = self.mse([data, ] + batch[1:], this_logsnr_broadcast, mse_type='epsilon', xinterval=xinterval).cpu()
                mses[-1][:, j] = this_mse

                if delta:
                    # MSE for estimator that rounds using x_hat
                    this_mse = self.mse([data, ] + batch[1:], this_logsnr_broadcast, mse_type='epsilon', xinterval=xinterval, delta=delta, soft=soft).cpu()
                    mses_round_xhat[-1][:, j] = this_mse

                    # Dequantize
                    this_mse = self.mse([data_dequantize] + batch[1:], this_logsnr_broadcast, mse_type='epsilon').cpu()
                    mses_dequantize[-1][:, j] = this_mse

        val_loss /= total_samples

        mses = t.cat(mses, dim=0)  # Concatenate the batches together across axis 0
        mses_round_xhat = t.cat(mses_round_xhat, dim=0)
        mses_dequantize = t.cat(mses_dequantize, dim=0)
        results['mses-all'] = mses  # Store array of mses for each sample, logsnr
        results['mses_round_xhat-all'] = mses_round_xhat
        results['mses_dequantize-all'] = mses_dequantize
        mses = mses.mean(dim=0)  # Average across samples, giving MMSE(logsnr)
        mses_round_xhat = mses_round_xhat.mean(dim=0)
        mses_dequantize = mses_dequantize.mean(dim=0)

        results['mses'] = mses
        results['mses_round_xhat'] = mses_round_xhat
        results['mses_dequantize'] = mses_dequantize
        results['mmse_g'] = self.mmse_g(logsnr.to(self.device)).to('cpu')

        results['nll (nats)'] = t.mean(self.h_g - 0.5 * w * t.clamp(results['mmse_g']  - mses, 0.))
        results['nll (nats) - dequantize'] = t.mean(self.h_g - 0.5 * w * t.clamp(results['mmse_g']  - mses_dequantize, 0.))
        results['nll (bpd)'] = results['nll (nats)'] / math.log(2) / self.d
        results['nll (bpd) - dequantize'] = results['nll (nats) - dequantize'] / math.log(2) / self.d

        if delta:
            results['nll-discrete-limit (bpd)'] = results['nll (bpd)'] - math.log(delta) / math.log(2.)
            results['nll-discrete-limit (bpd) - dequantize'] = results['nll (bpd) - dequantize'] - math.log(delta) / math.log(2.)
            if xinterval:  # Upper bound on direct estimate of -log P(x) for discrete x
                left_tail = 0.5 * t.log1p(t.exp(left_logsnr+self.log_eigs)).sum().cpu()
                j_max = int((xinterval[1] - xinterval[0]) / delta)
                right_tail = 4 * self.d * sum([math.exp(-(j-0.5)**2 * delta * delta * math.exp(right_logsnr)) for j in range(1, j_max+1)])
                mses_min = t.minimum(mses, mses_round_xhat)
                results['nll-discrete'] = t.mean(0.5 * (w * mses_min) + right_tail + left_tail)
                results['nll-discrete (bpd)'] = results['nll-discrete'] / math.log(2) / self.d

        ## Variance (of the mean) calculation - via CLT, it's the variance of the samples (over epsilon, x, logsnr) / n samples.
        ## n_samples is number of x samples * number of logsnr samples per x
        inds = (results['mmse_g']-results['mses']) > 0  # we only give nonzero estimates in this region (for continuous estimators)
        n_samples = results['mses-all'].numel()
        wp = w[inds]
        results['nll (nats) - var'] = t.var(0.5 * wp * (results['mmse_g'][inds] - results['mses-all'][:, inds])) / n_samples
        results['nll-discrete (nats) - var'] = t.var(0.5 * w * results['mses_round_xhat-all']) / n_samples  # Use entire range with discrete estimator
        results['nll (nats) - dequantize - var'] = t.var(0.5 * wp * (results['mmse_g'][inds] - results['mses_dequantize-all'][:, inds])) / n_samples
        results['nll (bpd) - std'] = t.sqrt(results['nll (nats) - var']) / math.log(2) / self.d
        results['nll (bpd) - dequantize - std'] = t.sqrt(results['nll (nats) - dequantize - var']) / math.log(2) / self.d
        results['nll-discrete (bpd) - std'] = t.sqrt(results['nll-discrete (nats) - var']) / math.log(2) / self.d

        return results, val_loss

    @property
    def loc_scale(self):
        """Return the parameters defining a normal distribution over logsnr, inferred from data statistics."""
        return self.loc_logsnr, self.scale_logsnr

    def dataset_info(self, dataloader, covariance_spectrum=None, diagonal=False):
        """covariance_spectrum can provide precomputed spectrum to speed up frequent experiments.
           diagonal: {False, True}  approximates covariance as diagonal, useful for very high-d data.
        """
        logger.log("Getting dataset statistics, including eigenvalues,")
        for batch in dataloader:
            break
        data = batch[0].to("cpu")  # assume iterator gives other things besides x in list
        logger.log('using # samples given:', len(data))
        self.d = len(data[0].flatten())
        if not diagonal:
            assert len(data) > self.d, f"Use a batch with more samples {len(data[0])} than dimensions {self.d}"
        self.shape = data[0].shape
        self.dtype = data[0].dtype
        self.left = (-1,) + (1,) * (len(self.shape))  # View for left multiplying a batch of samples
        x = data.flatten(start_dim=1)
        if covariance_spectrum:  # May save in cache to avoid processing for every experiment
            self.mu, self.U, self.log_eigs = covariance_spectrum
        else:
            var, self.mu = t.var_mean(x, 0)
            x = x - self.mu
            if diagonal:
                self.log_eigs = t.log(var)
                self.U = None  # "U" in this diagonal approximation should be identity, but would need to be handled specially to avoid storing large matrix.
            else:
                _, eigs, self.U = t.linalg.svd(x, full_matrices=False)  # U.T diag(eigs^2/(n-1)) U = covariance
                self.log_eigs = 2 * t.log(eigs) - math.log(len(x) - 1)  # Eigs of covariance are eigs**2/(n-1)  of SVD
            # t.save((self.mu, self.U, self.log_eigs), './covariance/cifar_covariance.pt')  # Save like this

        self.log_eigs = self.log_eigs.to(self.device)
        self.mu = self.mu.to(self.device)
        if self.U is not None:
            self.U = self.U.to(self.device)

        # Used to estimate good range for integration
        self.loc_logsnr = -self.log_eigs.mean().item()
        if diagonal:
            # A heuristic, since we won't get good variance estimate from diagonal - use loc/scale from CIFAR.
            self.loc_logsnr, self.scale_logsnr = 6.261363983154297, 3.0976245403289795
        else:
            self.scale_logsnr = t.sqrt(1+ 3. / math.pi * self.log_eigs.var()).item()

    def fit(self, dataloader_train, dataloader_test=None, epochs=10, use_optimizer='adam', lr=1e-4, verbose=False):
        """Given dataset, train the MMSE model for predicting the noise (or score).
           See image_datasets.py for example of torch dataset, that can be used with dataloader
           Shape needs to be compatible with model inputs.
        """
        if use_optimizer == 'adam':
            optimizer = t.optim.Adam(self.model.parameters(), lr=lr)
        elif use_optimizer == 'adamw':
            optimizer = t.optim.AdamW(self.model.parameters(), lr=lr)
        else:
            optimizer = t.optim.SGD(self.model.parameters(), lr=lr)

        # Return to standard fitting paradigm
        for i in range(1, epochs+1):  # Main training loop
            print("training ... ")
            train_loss = 0.
            t0 = time.time()
            total_samples = 0
            self.train()
            for batch in tqdm(dataloader_train):
                num_samples = len(batch[0])
                total_samples += num_samples
                optimizer.zero_grad()
                loss = self.nll(batch)
                loss.backward()
                optimizer.step()
                # Track running statistics
                train_loss += loss.detach().cpu().item() * num_samples
            train_loss /= total_samples
            iter_per_sec = len(dataloader_train) / (time.time() - t0)
            out_path = os.path.join(logger.get_dir(), f"model_epoch{i}.pt") 
            t.save(self.model.state_dict(), out_path) # save model
            self.logs['train loss'].append(train_loss)

            if dataloader_test:  # Process validation statistics once per epoch, if available
                print("testing ...")
                self.eval()
                with t.no_grad():
                    results, val_loss = self.test_nll(dataloader_test, npoints=100, delta=1./127.5, xinterval=(-1, 1), soft=True)
                    out_path = os.path.join(logger.get_dir(), f"results_epoch{i}.npy") 
                    np.save(out_path, results) 
                    self.logs['val loss'].append(val_loss)

            if verbose:
                if dataloader_test:
                    logger.log('epoch: {:3d}\t train loss: {:0.4f}\t val loss: {:0.4f}\t iter/sec: {:0.2f}'.
                        format(i, train_loss, val_loss, iter_per_sec))
                else:
                    logger.log('epoch: {:3d}\t train loss: {:0.4f}\t iter/sec: {:0.2f}'.
                        format(i, train_loss, iter_per_sec))

    @property
    def h_g(self):
        """Differential entropy for a N(mu, Sigma), where Sigma matches data, with same dimension as data."""
        return 0.5 * self.d * math.log(2 * math.pi * math.e) + 0.5 * self.log_eigs.sum().item()

    def mmse_g(self, logsnr):
        """The analytic MMSE for a Gaussian with the same eigenvalues as the data in a Gaussian noise channel."""
        return t.sigmoid(logsnr + self.log_eigs.view((-1, 1))).sum(axis=0)  # *logsnr integration, see note