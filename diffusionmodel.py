"""
Code for Diffusion model class
"""
import math
import time
import numpy as np
import torch as t
import torch.nn as nn
from tqdm import tqdm
from numpy import interp

from utilsiddpm import utils
from utilsiddpm.utils import logistic_integrate
from utilsiddpm import logger

class DiffusionModel(nn.Module):
    """Base class diffusion model for x, with optional conditional info, y.
       *logsnr integration: we do integrals in terms of logsnr instead of snr.
       Therefore we have to include a factor of 1/snr in integrands."""

    def __init__(self, model):
        super().__init__()
        self.model = model  # A model that takes in model(x, y (optional), snr, is_simple) and outputs the noise estimate
        self.logs = {"val loss": [], "train loss": [],
                     'mse_curves': [], 'mse_eps_curves': [],
                     'logsnr_loc': [], 'logsnr_scale': [],
                     'logsnr_grid': t.linspace(-6, 6, 100),  # Where to evaluate curves for later visualization
                    }  # store stuff for plotting, could use tensorboard for larger models
        self.loc_logsnr, self.scale_logsnr = 0., 2.  # initial location and scale for integration. Reset in data-driven way by dataset_info
        self.clip = 4 # initial quantile for integration.
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        print(f"Using {self.device} device for DiffusionModel")
        # dtype, dimensionality, and shape for data, set when we see it in "fit"
        self.dtype, self.d, self.shape, self.left = None, None, None, None

    def forward(self, batch, logsnr):
        """Batch is either [x,y] or [x,] depending on whether it is a conditional model."""
        return self.model(batch, t.exp(logsnr))

    def noisy_channel(self, x, logsnr):
        # TODO: fix incorrect broadcasting of noise if multiple x and one snr
        logsnr = logsnr.view(self.left)  # View for left multiplying
        eps = t.randn((len(logsnr),) + self.shape, dtype=x.dtype, device=x.device)
        return t.sqrt(t.sigmoid(logsnr)) * x + t.sqrt(t.sigmoid(-logsnr)) * eps, eps

    def mse(self, batch, logsnr, mse_type='epsilon', xinterval=None, delta=None, soft=False):
        """Return MSE curve either conditioned or not on y.
        x_hat = z/sqrt(snr) + eps_hat(z, snr)/sqrt(snr),
        so x_hat - x = (eps - eps_hat(z, snr))/sqrt(snr).
        And we actually reparametrize eps_hat to depend on eps(z/sqrt(1+snr), snr)
        Options are:
        mse_type: {"epsilon" or "x"), for error in predicting noise, or predicting data.
        xinterval: Whether predictions should be clamped to some interval.
        delta: If provided, round to the nearest discrete value
        """
        x = batch[0].to(self.device)  # assume iterator gives other things besides x in list
        z, eps = self.noisy_channel(x, logsnr)
        noisy_batch = [z, ] + batch[1:]  # batch may include conditioning on y
        eps_hat = self.model(noisy_batch, t.exp(logsnr))
        x_hat = t.sqrt(1 + t.exp(-logsnr.view(self.left))) * z - eps_hat * t.exp(-logsnr.view(self.left) / 2)
        if delta:
            if soft:
                x_hat = utils.soft_round(x_hat, t.exp(logsnr).view(self.left), xinterval, delta)
            else:
                x_hat = delta * t.round((x_hat - xinterval[0]) / delta) + xinterval[0]  # Round to nearest discrete value
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
        return loss  # *logsnr integration, see note

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
                    this_mse = self.mse([data, ] + batch[1:], this_logsnr_broadcast, mse_type='epsilon').cpu()
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

        # Variance (of the mean) calculation - via CLT, it's the variance of the samples (over epsilon, x, logsnr) / n samples.
        # n_samples is number of x samples * number of logsnr samples per x
        inds = (results['mmse_g']-results['mses']) > 0  # we only give nonzero estimates in this region (for continuous estimators)
        n_samples = results['mses-all'].numel()
        wp = w[inds].to('cpu')
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
        print("Getting dataset statistics, including eigenvalues,")
        for batch in dataloader:
            break
        data = batch[0].to("cpu")  # assume iterator gives other things besides x in list
        print('using # samples given:', len(data))
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
            # t.save((self.mu, self.U, self.log_eigs), './scripts/imagenet64_covariance.pt')  # Save like this

        self.logs['log_eigs'] = self.log_eigs.cpu()
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

    def fit(self, dataloader_train, dataloader_test=None, epochs=10, use_optimizer='adam', lr=1e-4, verbose=False, iddpm=False):
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
        for i in range(1, epochs+1):  # Main training loop
            ### We can try various 'lr_scheduler' here ### 
            # lr decay -- multistep
            # if i == 5:
            #     for p in optimizer.param_groups:
            #         p['lr'] *= 0.1

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
            if not verbose:
              logger.log("epoch: {:3d}\t train loss: {:0.4f}".format(i, train_loss/np.log(2.0)/self.d))
            if iddpm:
                t.save(self.model.state_dict(), f'/media/theo/Data/checkpoints/iid_sampler/iddpm/model_epoch{i}.pt') # save model
            else:
                t.save(self.model.state_dict(), f'/media/theo/Data/checkpoints/iid_sampler/ddpm/model_epoch{i}.pt') # save model
            self.log_function(train_loss=train_loss)

            if dataloader_test:  # Process validation statistics once per epoch, if available
                print("testing ...")
                self.eval()
                with t.no_grad():
                    results, val_loss = self.test_nll(dataloader_test, npoints=100, delta=1./127.5, xinterval=(-1, 1))
                    if iddpm:
                        np.save(f"/home/theo/Research_Results/debug/iid_sampler/iddpm/results_epoch{i}_base.npy", results) # save test results
                    else:
                        np.save(f"/home/theo/Research_Results/debug/iid_sampler/ddpm/results_epoch{i}_base.npy", results) # save test results
                    self.log_function(val_loss=val_loss, results=results)

            if verbose:
                logger.log('epoch: {:3d}\t train loss: {:0.4f}\t val loss: {:0.4f}\t iter/sec: {:0.2f}'.
                      format(i, train_loss, val_loss, iter_per_sec))
                logger.log('nll (nats): {:0.4f}\t nll-discrete (bpd): {:0.4f}\t nll-discrete-limit (bpd): {:0.4f}\t nll-discrete-limit (bpd) - dequantize:{:0.4f}'.
                    format(results['nll (nats)'],
                        results['nll-discrete (bpd)'], 
                        results['nll-discrete-limit (bpd)'], 
                        results['nll-discrete-limit (bpd) - dequantize']))

    def log_function(self, train_loss=None, val_loss=None, results=None):
        """Record logs during training."""
        self.logs['logsnr_loc'].append(self.loc_logsnr)
        self.logs['logsnr_scale'].append(self.scale_logsnr)
        if train_loss:
            self.logs['train loss'].append(train_loss)
        if val_loss:
            self.logs['val loss'].append(val_loss)
        if results:
            self.logs['mse_curves'].append(results['mses'] / t.exp(self.logs['logsnr_grid']))
            self.logs['mse_eps_curves'].append(results['mses'])


    ##########################################
    #         SAMPLING RELATED METHODS       #
    ##########################################
    @t.no_grad()
    def sample(self, schedule, n_samples=1, info_init=False, m=1., temp=1., store_t=False, verbose=True, precondition=False):
        """Generate samples from noise
        schedule - logsnr values for Langevin steps
        n_samples - number to generate
        info_init - whether to start with standard normal or normal with same spectrum - may require schedule change
        temp - can control temperature, i.e. sample with a distribution raised to some power. temp=0 seeks modes
        store_t = integer or False - output every k-th step
        m = 1  #  Mystery factor. Should be 1 according to derivation, much better with 2.
        """
        n_steps = len(schedule) - 1
        schedule = schedule.type(self.dtype)  # Match data type
        z = self.sample_g(n_samples, match_cov=info_init, match_mu=info_init)
        if store_t:  # Store outputs on CPU, optionally over time
            zs = t.empty((n_samples, 1 + n_steps // store_t) + self.shape, device='cpu', dtype=self.dtype)
            zs[:, 0] = z.cpu()

        for tt in range(len(schedule) - 1):  # Langevin dynamics loop
            snr = t.exp(schedule[tt])
            snr_target = t.exp(schedule[tt + 1])
            assert snr_target >= snr, "Monotonically increase SNR in schedule"
            # Old strategy
            # eta = 2 * (m * snr_target - snr) / (m * snr_target * (1 + snr))
            # New strategy - m=1 seems fine
            min_eig = t.exp(self.log_eigs)[-1]  # Minimum variance of Gaussian fitting data
            min_var_alpha = (snr * min_eig + 1) / (snr + 1)  # Smallest var at some snr
            eta = m * min_var_alpha  # This way the std of noise will not be bigger than smallest std of data
            if verbose:  # SLOW: turn off for practical runs
                print("snr: {:0.2f}, eta: {:0.5f}, -log p(x): {:0.2f}".format(snr, eta, self.nll([z])))

            # Langevin step
            snr_cast = snr * t.ones(len(z), dtype=z.dtype, device=z.device)
            delta = -eta / 2 * t.sqrt(1 + snr) * self.model([z], snr_cast)  # Deterministic part of Langevin
            noise = self.sample_g(n_samples, match_cov=precondition, snr=snr)
            if precondition:  # From HMC perspective, Kinetic energy = 1/2 v^T Sigma v, where Sigma is data cov.
                delta = self.mult_cov(delta, snr)
            z += delta + temp * t.sqrt(eta) * noise

            if store_t and (tt + 1) % store_t == 0:
                zs[:, (tt + 1) // store_t] = z.cpu()
        if not store_t:
            zs = z.cpu()  # Return final samples only
        return zs

    @t.no_grad()
    def outlier_score(self, data, npoints=1000):
        logsnr, w = logistic_integrate(npoints=npoints, loc=7., scale=5., clip=2., device=self.device, deterministic=True)
        scores = [self.mmse_g_x(x, logsnr) - self.mse([x], logsnr) for x in data]
        return [(w * s).mean() for s in scores]

    @t.no_grad()
    def sample_cold(self, schedule, n_samples=1, store_t=False, verbose=True):
        """Generate samples from noise using cold diffusion paper idea
        schedule - logsnr values for Langevin steps
        n_samples - number to generate
        info_init - whether to start with standard normal or normal with same spectrum - may require schedule change
        temp - can control temperature, i.e. sample with a distribution raised to some power. temp=0 seeks modes
        store_t = integer or False - output every k-th step
        m = 1  #  Mystery factor. Should be 1 according to derivation, much better with 2.
        """
        n_steps = len(schedule) - 1
        schedule = schedule.type(self.dtype)  # Match data type
        z = t.randn((n_samples,) + self.shape, dtype=self.dtype, device=self.device)
        if store_t:  # Store outputs on CPU, optionally over time
            zs = t.empty((n_samples, 1 + n_steps // store_t) + self.shape, device='cpu', dtype=self.dtype)
            zs[:, 0] = z.cpu()

        for tt in range(len(schedule) - 1):
            logsnr = schedule[tt]
            snr = t.exp(logsnr)
            assert schedule[tt + 1] >= schedule[tt], "Monotonically increase SNR in schedule"
            snr_cast = snr * t.ones(len(z), dtype=z.dtype, device=z.device)
            xhat = t.sqrt(1 + t.exp(-logsnr)) * z - self.model([z], snr_cast) * t.exp(-logsnr / 2)
            z2, _ = self.noisy_channel(xhat, schedule[tt+1])
            z1, _ = self.noisy_channel(xhat, schedule[tt])
            z = z2  # z - z1 + z2
            # super(type(model), model).forward(x, t.tensor([0.,] * 5, device='cuda'))


            if store_t and (tt + 1) % store_t == 0:
                zs[:, (tt + 1) // store_t] = z.cpu()
        if not store_t:
            zs = z.cpu()  # Return final samples only
        return zs

    def sample_bp(self, npoints=100, n_steps=1000, eta=0.001, temp=1., store_t=False, verbose=True):
        """Generate 1 sample from noise
        npoints - to use in approximating NLL
        info_init - whether to start with standard normal or normal with same spectrum - may require schedule change
        temp - can control temperature, i.e. sample with a distribution raised to some power. temp=0 seeks modes
        store_t = integer or False - output every k-th step
        """
        n_samples = 1  # Requires re-write to do in parallel, but not feasible anyway because backprop on sum requries a lot of memory
        z = t.randn((n_samples,) + self.shape, dtype=self.dtype, device=self.device)
        # z = z - self.model([z], t.tensor([0.]))
        z = t.autograd.Variable(z, requires_grad=True)

        if store_t:  # Store outputs on CPU, optionally over time
            zs = t.empty((n_samples, 1 + n_steps // store_t) + self.shape, device='cpu', dtype=self.dtype)
            zs[:, 0] = z.detach().cpu()

        for tt in range(n_steps):  # Langevin dynamics loop
            # if verbose:  # SLOW: turn off for practical runs
            #     print("-log p(x): {:0.2f}".format(self.nll([z], logsnr_samples_per_x=300)))

            # Langevin step
            grad_nll = 0.
            av_mse = 0.
            num_repeat = 100
            for k in range(num_repeat):
                logsnr, w = logistic_integrate(npoints, loc=6., scale=5.5, clip=1., device=self.device)
                mse = (w * self.mse([z], logsnr, mse_type='epsilon')).mean()
                grad_nll += 0.5 * t.autograd.grad(mse, [z])[0].detach() / num_repeat
                av_mse = mse.detach() / num_repeat
            print('mse', av_mse.item())

            c = 2 - 1. / (1. + math.exp(-5)) + 1. / (1. + math.exp(17))  # endpoints of integration using loc=6, scale=5.5., clip=2 above
            grad_nll += z.data * c  # add part for tails

            z.data = z.data - eta * eta / 2 * grad_nll + temp * eta * t.randn_like(z)
            # with t.no_grad():
            #     nll0 = self.nll([z], logsnr_samples_per_x=100)
            #     z_prop = z.data - eta * eta / 2 * grad_nll + temp * eta * t.randn_like(z)
            #     nll1 = self.nll([z_prop], logsnr_samples_per_x=100)
            # if nll1 < nll0:
            #     z.data = z_prop
            # else:
            #     print(nll0, nll1, 'reject')

            if store_t and (tt + 1) % store_t == 0:
                zs[:, (tt + 1) // store_t] = z.detach().cpu()
        if not store_t:
            zs = z.cpu()  # Return final samples only
        return zs

    def e_z0_given_z1(self, z1, snr1, snr_del):
        """Assume we have a markov chain x - z0 - z1. p(z0|x) is Gaussian, variance preserving
        with logsnr0. z0 to z1 is Gaussian, variance preserving with logsnr_del.
        Estimate the expected value of z0, given z1, in terms of our model."""
        c1 = t.sqrt(1 + 1 / snr_del)
        c0 = t.sqrt((1+snr1) / (snr_del * (1 + snr_del)))
        c0, c1 = c0.view(self.left), c1.view(self.left)
        return c1 * z1 - c0 * self.model([z1], snr1 * t.ones(len(z1), dtype=z1.dtype, device=z1.device))

    @t.no_grad()
    def generate_schedule(self, npoints=500, dataloader=None, info_init=False, batch_size=100, plot_grid=False):
        """Generate a schedule of logsnrs for sampling.
        If test_data is no provided, we use the data covariance to estimate this, though it is
        unlikely to be as effective.
        info_init uses the non-Gaussian info only, maybe compatible with info_init in sample method.
        """
        logsnr, w = logistic_integrate(npoints + 1, *self.loc_scale, device=self.device, clip=5)

        if dataloader is None:
            assert not info_init, "informative initialization requires data to estimate (non-gaussian) info gain"
            gaps = self.mmse_g_1(logsnr) - self.mmse_g(logsnr)
        else:
            mses = t.zeros(npoints + 1, device=self.device)

            for ii, batch in enumerate(dataloader):
                batch = [batch[0].to(self.device)]
                for j, this_logsnr in enumerate(logsnr):
                    this_logsnr_broadcast = this_logsnr * t.ones(len(batch[0]), device=batch[0].device)
                    this_mse = t.mean(self.mse(batch, this_logsnr_broadcast, mse_type='epsilon'))
                    mses[j] += this_mse
            mses /= len(dataloader)  # Average over batches
            if info_init:
                gaps = self.mmse_g(logsnr) - mses
            else:
                gaps = self.mmse_g_1(logsnr) - mses
            gaps = t.clamp(gaps, 0.)  # If gap is negative, we pick Gaussian decoder instead which has zero gap

        fig = (w * gaps).cumsum(0) / (w * gaps).sum()  # Fractional Information Gap (FIG(snr))
        info_grid = t.linspace(0., 1., npoints + 3, device=fig.device)[1:-1]
        schedule = t.from_numpy(interp(info_grid.cpu().numpy(), fig.cpu().numpy(), logsnr.cpu().numpy())).to(fig.device)

        if plot_grid:  # code to visualize grid
            import matplotlib.pyplot as plt
            inv = interp(schedule.cpu().numpy(), logsnr.cpu().numpy(), fig.cpu().numpy())
            f, axs = plt.subplots(2, sharex=True)
            ax = axs[0]
            ax.set_xlabel('log SNR')
            ax.set_ylabel('MSE Gap')
            ax.plot(logsnr.cpu(), gaps.cpu(), color='black', linewidth=3)

            ax = axs[1]
            colors = plt.cm.plasma(t.linspace(0, 1, len(logsnr)))
            inv = interp(schedule.cpu().numpy(), logsnr.cpu().numpy(), fig.cpu().numpy())
            ax.plot(logsnr.cpu(), fig.cpu(), color='black', linewidth=3)
            ax.hlines(info_grid.cpu(), logsnr[0].item(), schedule.cpu(), colors=colors)
            ax.vlines(schedule.cpu(), 0, inv, colors=colors)
            ax.set_xlabel('log SNR')
            ax.set_ylabel('FIG(SNR)')

            f.savefig('/home/gregv/diffusion/figures/info_gain.png')

        return schedule


    ############################################
    #     Methods for base model, Gaussian     #
    ############################################
    def sample_g(self, n_samples=1, match_cov=True, match_mu=False, snr=False):
        """Generate a sample from the Gaussian with the same mean/covariance as data"""
        z = t.randn((n_samples, self.d), dtype=self.dtype, device=self.device)
        if match_cov:
            if snr:
                lam = t.sqrt(t.exp(self.log_eigs) * snr / (1+snr) + 1/(1+snr))
            else:
                lam = t.sqrt(t.exp(self.log_eigs))  # snr->infinity limit
            z = t.einsum('ij,j,jk,lk->li', self.U.t(), lam, self.U, z)
        if match_mu:
            mu = self.mu
        else:
            mu = 0.
        return (z + mu).view((-1,) + self.shape)

    def mult_cov(self, x, snr):
        """Multiple samples, x, by the covariance matrix"""
        x = x.flatten(start_dim=1)
        lam = t.exp(self.log_eigs) * snr / (1+snr) + 1./(1.+snr)
        return t.einsum('ij,j,jk,lk->li', self.U.t(), lam, self.U, x).view((-1,) + self.shape)

    def nll_g(self, x):
        """ -log p_G(x) for a single input, x"""
        x = x.view((-1, self.d)) - self.mu
        energy = 0.5 * t.einsum('li,ij,j,jk,lk->l', x, self.U.t(), t.exp(-self.log_eigs), self.U, x)
        logZ = self.d / 2 * math.log(2 * math.pi) + 0.5 * self.log_eigs.sum().item()
        return energy + logZ

    @property
    def h_g(self):
        """Differential entropy for a N(mu, Sigma), where Sigma matches data, with same dimension as data."""
        return 0.5 * self.d * math.log(2 * math.pi * math.e) + 0.5 * self.log_eigs.sum().item()

    def mmse_g(self, logsnr):
        """The analytic MMSE for a Gaussian with the same eigenvalues as the data in a Gaussian noise channel."""
        return t.sigmoid(logsnr + self.log_eigs.view((-1, 1))).sum(axis=0)  # *logsnr integration, see note

    def mmse_g_1(self, logsnr):
        """The analytic MMSE for a standard Gaussian in a Gaussian noise channel."""
        return self.d * t.sigmoid(logsnr)  # *logsnr integration, see note

    def mmse_g_x(self, x, logsnr):
        """The analytic pointwise MMSE, mmse_G(x, snr), for Gaussian with same spectrum as data.
           Assume x is a single point and logsnr is a list."""
        logsnr = logsnr.view((-1, 1))
        x = x.view((-1, self.d)) - self.mu
        diag = 1. / t.square(t.exp(logsnr + self.log_eigs) + 1)
        A = t.einsum('li,ij,rj,jk,lk->lr', x, self.U.t(), diag, self.U, x)
        B = (t.exp(logsnr) / t.square(t.exp(logsnr) + t.exp(-self.log_eigs))).sum(dim=1)
        return A + B

    def grad_g(self, x):
        """The analytic gradient of -log p(x) for a Gaussian with the same mean/covariance as data."""
        # sig_inv = t.mm(t.mm(self.U.t(), t.diag(t.exp(-self.log_eigs))), self.U)
        x = x.flatten() - self.mu
        return t.einsum('ij,j,jk,k->i', self.U.t(), t.exp(-self.log_eigs), self.U, x).view(self.shape)
        # return t.matmul(x, sig_inv).view(self.shape)

    def grad_mmse_g(self, x, logsnr):
        """The analytic gradient of 1/2 mmse(x,snr) for a Gaussian with the same mean/covariance as data.
        For a single x, but a list of logsnr."""
        a = 1. / t.square(t.exp(logsnr.view((-1, 1)) + self.log_eigs) + 1)
        x = x.flatten() - self.mu
        grad = t.einsum('ij,rj,jk,k,r->rk', self.U.t(), a, self.U, x, t.exp(logsnr))
        return grad.view((-1,) + self.shape)  # *logsnr integration, see note


    #################################
    #      Things to implement      #
    #################################
    def mi(self, x, y):
        """Estimate the MI from the scores, for data from the fitted model."""
        # Do... many snr for each x, y pair
        # Then do the same, but with y's chosen randomly.
        raise NotImplentedError
        with t.no_grad():
            self.mse(x, y, snr)