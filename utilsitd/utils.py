# Setup, initialization
import torch as t
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.style.use('seaborn-paper')  # also try 'seaborn-talk', 'fivethirtyeight'


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


def soft_round(x, snr, xinterval, delta):
    ndim = len(x.shape)
    bins = t.linspace(xinterval[0], xinterval[1], 1 + int((xinterval[1]-xinterval[0])/delta), device=x.device)
    bins = bins.reshape((-1,) + (1,) * ndim)
    ps = t.nn.functional.softmax(-0.5 * t.square(x - bins) * (1 + snr), dim=0)
    return (bins * ps).sum(dim=0)

