# Setup, initialization
import torch as t
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.style.use('seaborn-paper')  # also try 'seaborn-talk', 'fivethirtyeight'


class CustomDataset(t.utils.data.Dataset):
    """A simple dataset constructor, that optionally includes side information, y,
       In a format to use with pytorch dataloaders.
    """
    def __init__(self, x, y=None):
        self.data = x
        self.y = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # (self.x[idx] - self.mean) / self.std
        if self.y is None:
            return (x,)
        else:
            y = self.y[idx]
            return x, y


def viz(logs, d=2):
    """Default visualization for logs"""
    fig, axs = plt.subplots(3, 1, sharex=False, sharey=False)
    epochs = len(logs['val loss'])
    niter = len(logs['train loss'])
    ax = axs[0]
    ax.plot(np.arange(niter) * epochs / niter, logs['train loss'], label='train loss')
    ax.plot(np.arange(epochs), logs['val loss'], label='val loss')
    # ax.set_yscale('log')
    ax.legend()

    ax.set_ylabel('Training Loss')
    ax.set_xlabel('Epochs')

    logsnrs = logs['logsnr_grid']
    snrs = t.exp(logsnrs)
    mses = logs['mse_eps_curves']

    baseline = np.array([d / (1. + np.exp(-logsnr)) for logsnr in logsnrs])
    baseline2 = t.sigmoid(logsnrs + logs['log_eigs'].view((-1, 1))).sum(axis=0).numpy()
    ax = axs[1]
    ax.plot(logsnrs, baseline, label='$N(0,1)$ MMSE')
    ax.plot(logsnrs, baseline2, lw=3, label='$N(\mu,\Sigma)$ MMSE')
    ax.plot(logsnrs, mses[-1], label='Data MSE')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\gamma$)')
    #ax.set_xscale('log')
    ax.legend()

    ax = axs[2]
    ax.plot(logsnrs, baseline - np.array(mses[-1]), label='MMSE Gap for $N(0,1)$')
    ax.plot(logsnrs, baseline2 - np.array(mses[-1]), label='MMSE Gap for $N(\mu,\Sigma)$')
    ax.set_ylim(-0.01, np.max(np.array(baseline) - np.array(mses[-1])))
    ax.set_ylabel('MMSE Gap $(\epsilon)$')
    ax.set_xlabel('log SNR ($\gamma$)')
    ax.legend()
    #ax.set_xscale('log')

    fig.set_tight_layout(True)
    return fig


def trunc_normal_integrate(npoints, loc, scale, clip=3, device='cpu'):
    """Monte Carlo integration with importance sampling, using a truncated normal as the base distribution
    \int f(a) da = \int f(a) p(a)/p(a) da = E_{p(a)} [f(a) * w(a)], where w(a) = 1/p(a).
    We draw samples a ~ p(a) and ase a Monte Carlo estimator for expectation, and return the appropriate weights.
    The base distribution is a truncated normal with N(loc, scale**2) and clipping "clip" stds from the mean.
    """
    a = t.empty(npoints, device=device)
    loc, scale, clip = t.tensor(loc, device=device), t.tensor(scale, device=device), t.tensor(clip, device=device)
    clip_a = loc - clip * scale
    clip_b = loc + clip * scale
    t.nn.init.trunc_normal_(a, loc, scale, a=clip_a, b=clip_b)
    w = t.exp(0.5 * t.square(a - loc) / scale**2) * t.sqrt(2. * np.pi * scale**2) * t.erf(clip / np.sqrt(2.))
    return a, w


def logistic_integrate(npoints, loc, scale, clip=4., device='cpu', deterministic=False):
    """Return sample point and weights for integration, using
    a truncated logistic distribution as the base, and importance weights.
    """
    loc, scale, clip = t.tensor(loc, device=device), t.tensor(scale, device=device), t.tensor(clip, device=device)

    # IID samples from uniform, use inverse CDF to transform to target distribution
    if deterministic:
        t.manual_seed(0)
    ps = t.rand(npoints, dtype=loc.dtype, device=device)
    ps = t.sigmoid(-clip) + (t.sigmoid(clip) - t.sigmoid(-clip)) * ps  # Scale quantiles to clip
    logsnr = loc + scale * t.logit(ps)  # Using quantile function for logistic distribution

    # importance weights
    weights = scale * t.tanh(clip / 2) / (t.sigmoid((logsnr - loc)/scale) * t.sigmoid(-(logsnr - loc)/scale))
    return logsnr, weights


def trunc_inv_integrate(npoints, loc, scale, clip=4., device='cpu', deterministic=False, eps=0.):
    """Return sample point and weights for integration, using
    a truncated distribution proportional to 1 / (1+snr) as the base, and importance weights.
    loc, scale, clip  - are same as for continuous density estimator, just used to fix the range
    parameter, eps=1 is the form implied by optimal Gaussian MMSE at low SNR.
    True MMSE drops faster, so we use a smaller constant
    """
    loc, scale, clip = t.tensor(loc, device=device), t.tensor(scale, device=device), t.tensor(clip, device=device)
    left_snr, right_snr = t.exp(loc - clip * scale), t.exp(loc + clip * scale)  # truncated range

    # IID samples from uniform, use inverse CDF to transform to target distribution
    if deterministic:
        t.manual_seed(0)
    ps = t.rand(npoints, dtype=loc.dtype, device=device)
    Z = t.log(eps + right_snr) - t.log(eps + left_snr)  # normalization constant
    snr = (eps + left_snr) * t.exp(Z * ps) - eps  # Use quantile function

    # importance weights
    weights = Z * (eps + snr)
    return snr, weights


def plot_image(img_coll, num_samples, logsnrs):
    fig, axs = plt.subplots(num_samples*len(logsnrs), len(img_coll), sharex=False, sharey=False, figsize=(10,24))

    for i in range(num_samples*len(logsnrs)):
        for j, k in enumerate(img_coll.keys()):
            ax = axs[i][j]
            if i == 0:
                ax.set_title(k)
            if j == 0:
                ax.set_ylabel("$\\alpha$: "+str(logsnrs[i%len(logsnrs)]))
            ax.imshow(np.transpose(img_coll[k][i][0], [1,2,0]))
            ax.tick_params(labelbottom=False, labelleft=False)
    fig.set_tight_layout(True)
    plt.show()


def soft_round(x, snr, xinterval, delta):
    ndim = len(x.shape)
    bins = t.linspace(xinterval[0], xinterval[1], 1 + int((xinterval[1]-xinterval[0])/delta), device=x.device)
    bins = bins.reshape((-1,) + (1,) * ndim)
    ps = t.nn.functional.softmax(-0.5 * t.square(x - bins) * (1 + snr), dim=0)
    return (bins * ps).sum(dim=0)


def viz_soft_round():
    z = t.linspace(-2, 2, 10000)
    delta = 0.25
    xrange = (-1, 1)

    fig, ax = plt.subplots(1,1)

    for s in [1, 100, 1000]:
        ax.plot(z, soft_round(z, s, xrange, delta), label='SNR={}'.format(s))

    # Major ticks every 20, minor ticks every 5
    minor_ticks_y = np.arange(xrange[0], xrange[1]+delta, delta)
    minor_ticks_x = np.arange(-2, 2+delta, delta)

    major_ticks_y = minor_ticks_y[::4]
    major_ticks_x = minor_ticks_x[::4]

    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax.set_aspect(0.5)
    ax.set_xlabel('$z$')
    ax.set_ylabel('soft-disc($z$, SNR)')
    ax.grid(which='both', linestyle='--', linewidth=0.3)
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig('soft_round_plot.pdf',bbox_inches='tight')
