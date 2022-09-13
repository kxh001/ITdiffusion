""" Architectures
TODO: SimpleResNet  (non-images) adapted from
https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html
Specifically, https://github.com/Yura52/rtdl/blob/main/rtdl/modules.py
TODO: Eventually test with images using U-Net from GLIDE, Dall-e-2, imagen etc.
"""
import math
import enum
from scipy import linalg
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import torch as t
import torch.nn as nn
from torch import Tensor


ModuleType = Union[str, Callable[..., nn.Module]]

def exists(x):
    return x is not None

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            return ReGLU()
        elif module_type == 'GEGLU':
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
            return cls(*args)
    else:
        return module_type(*args)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = t.exp(
        -math.log(max_period) * t.arange(start=0, end=half, dtype=t.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = t.cat([t.cos(args), t.sin(args)], dim=-1)
    if dim % 2:
        embedding = t.cat([embedding, t.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def Upsample(dim_in, dim_out):
    return nn.Linear(dim_in, dim_out) 

def Downsample(dim_in, dim_out):
    return nn.Linear(dim_in, dim_out) 

class LinNet(nn.Module):
    def __init__(self, dim_x=2, dim_y=0, dim_embed=16):
        super(LinNet, self).__init__()
        self.flatten = nn.Flatten()
        self.dim = dim_x + dim_y
        self.time_dim = self.dim * 4
        self.fc = nn.Linear(self.dim + self.time_dim, dim_x)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

    def forward(self, batch, snr, is_simple=False):
        x = self.flatten(batch[0])
        assert len(x) == len(snr)
        if is_simple:
            snr = timestep_embedding(snr, self.time_dim)
        else:
            snr = timestep_embedding(snr, self.dim) 
            snr = self.time_mlp(snr)
        if len(batch) > 1:
            y = self.flatten(batch[1])
            assert len(y) == len(x)
            inputs = t.cat((x, y, snr), 1)
        else:
            inputs = t.cat((x, snr), 1)
        return self.fc(inputs)


class ResNet(nn.Module):
    """The ResNet model used in [gorishniy2021revisiting].
    The following scheme describes the architecture:
    .. code-block:: text
        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)
                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)
          Head: (in) -> Norm -> Activation -> Linear -> (out)
    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )
            assert module(x).shape == (len(x), 1)
    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor, scale_shift=None) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)

            if exists(scale_shift):
                scale, shift = scale_shift
                x = x * (scale + 1) + shift

            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        if d_main is None:
            d_main = d_in
        self.first_layer = nn.Linear(d_in, d_main)
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
        cls: Type['ResNet'],
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        d_out: int,
    ) -> 'ResNet':
        """Create a "baseline" `ResNet`.
        This variation of ResNet was used in [gorishniy2021revisiting]. Features:
        * :code:`Activation` = :code:`ReLU`
        * :code:`Norm` = :code:`BatchNorm1d`
        Args:
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.
        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor, time_emb=None) -> Tensor:
        x = self.first_layer(x)
        if exists(time_emb): 
            scale_shift = time_emb.chunk(2, dim = 1) # dim*2
            x = self.blocks(x, scale_shift = scale_shift) # d_main * d_main
        else:
            x = self.blocks(x)
        x = self.head(x)
        return x


class SimpleResNet(nn.Module):
    def __init__(self, dim_x=2, dim_y=0, dim_embed=16):
        super(SimpleResNet, self).__init__()
        self.flatten = nn.Flatten()
        self.dim_dt = dim_x + dim_y
        self.time_dim = self.dim_dt * 4

        self.resnet = ResNet.make_baseline(
            d_in=self.dim_dt + self.time_dim, 
            n_blocks=2,
            d_main=2*(self.dim_dt + self.time_dim), 
            d_hidden=4*(self.dim_dt + self.time_dim), 
            dropout_first=0.25, # default 0.25
            dropout_second=0.0, # default 0.0
            d_out=dim_x
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim_dt, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

    def forward(self, batch, snr, is_simple=False):
        x = self.flatten(batch[0])
        assert len(x) == len(snr)
        if is_simple:
            snr = timestep_embedding(snr, self.time_dim)
        else:
            snr = timestep_embedding(snr, self.dim_dt) 
            snr = self.time_mlp(snr)
        if len(batch) > 1:
            y = self.flatten(batch[1])
            assert len(y) == len(x)
            inputs = t.cat((x, y, snr), 1)
        else:
            inputs = t.cat((x, snr), 1)
        return self.resnet(inputs)


class GaussTrueMMSE(nn.Module):
    """For testing, we construct several ground truth quantities for a Gaussian, N(0, Sigma).
       Calling the (forward) module computes the optimal eps_hat(z, snr),
       for the optimal denoising estimator x_hat, which is written as,
       x_hat(z, snr) = (z - eps_hat)/sqrt(snr)   # to match Diffusion literature conventions
    """
    def __init__(self, cov, device):
        super().__init__()
        self.cov = t.from_numpy(cov).type(t.FloatTensor)
        self.cov = self.cov.to(device)
        self.d = len(self.cov)
        self.S, self.U = t.linalg.eigh(self.cov)
        self.prec = t.mm(t.mm(self.U, t.diag(1. / self.S)), self.U.t())  # Precision is inverse covariance
        self.logZ = 0.5 * self.d * math.log(2 * math.pi) + 0.5 * t.log(self.S).sum()  # Log normalization
        self.dummy = t.nn.Parameter(t.randn(2))
        self.register_parameter(name='dummy', param=self.dummy)  # Dummy parameter so it runs fit method

    def entropy(self):
        """Differential entropy for a Gaussian"""
        return self.logZ + self.d / 2

    def mmse(self, snr):
        """Minimum mean square error at a given SNR level."""
        return (1. / (1. / self.S + snr.view((-1, 1)))).sum(axis=1)

    def nll(self, x):
        """-log p(x)"""
        return 0.5 * t.mm(t.mm(x, self.prec), x.t()) + self.logZ

    def true_grad(self, x):
        """- \nabla_x log p(x)"""
        return t.matmul(x, self.prec)

    def true_grad_mmse(self, x, snr):
        """$- \nabla_x 1/2 mmse(x, snr)$"""
        a = 1. / t.square(snr * self.S + 1)
        M = t.mm(t.mm(self.U, t.diag(a)), self.U.t())
        return t.matmul(x, M)

    def forward(self, batch, snr):
        """The DiffusionModel expects to get:
        eps_hat(z, snr), where z = sqrt(snr/(1+snr) x + sqrt(1/(1+snr) eps,
        and x_hat = sqrt((1+snr)/snr) * z - eps_hat / sqrt(snr)
        For Gaussians, we derive the optimal estimator:
        x_hat^* = sqrt(snr/(1+snr)) (snr/(1+snr) I + Sigma^-1/(1+snr))^-1 z
        The matrix inverses we handle with the precomputed SVD of Sigma (covariance for x).
        """
        z = batch[0]  # Noisified batch, z = sqrt(snr/(1+snr)) x + sqrt(1/(1+snr)) * eps, eps ~ N(0,I)
        assert len(z) == len(snr)
        snr = snr.view((-1, 1))
        xhat = t.mm(z, self.U)
        xhat = ((1. + snr) / (snr + 1. / self.S)) * xhat
        xhat = t.mm(xhat, self.U.t())
        xhat = t.sqrt(snr / (1. + snr)) * xhat
        # Now return eps_hat estimator
        return t.sqrt(1+snr) * z - t.sqrt(snr) * xhat + 0. * self.dummy.sum()  # Have to include dummy param in forward


class TabUnet(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y = 0,
        dim_mults = (2, 4, 8, 8), # (2,4,8,8)
    ):
        super().__init__()

        # determine dimensions

        self.dim_dt = dim_x + dim_y
        dims = [self.dim_dt, *map(lambda m: self.dim_dt // m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        self.time_dim = self.dim_dt * 4

        # layers

        self.flatten = nn.Flatten()
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResNet.make_baseline(
                    d_in=dim_in,
                    n_blocks=1,
                    d_main=self.dim_dt*2,
                    d_hidden=self.dim_dt*2, # align to scale & shift
                    dropout_first=0.25,
                    dropout_second=0.0,
                    d_out=dim_in
                ),
                Downsample(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block = ResNet.make_baseline(
                    d_in=mid_dim,
                    n_blocks=1,
                    d_main=self.dim_dt*2,
                    d_hidden=self.dim_dt*2, # align to scale & shift
                    dropout_first=0.25,
                    dropout_second=0.0,
                    d_out=mid_dim
                )
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[:-1])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResNet.make_baseline(
                    d_in=dim_out*2,
                    n_blocks=1,
                    d_main=self.dim_dt*2,
                    d_hidden=self.dim_dt*2, # align to scale & shift
                    dropout_first=0.25,
                    dropout_second=0.0,
                    d_out=dim_out*2
                ),
                Upsample(dim_out*2, dim_in) if not is_last else nn.Identity()
            ]))

        self.final_layer = nn.Linear(self.dim_dt, dim_x)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim_dt, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

    def forward(self, batch, snr, is_simple=False):
        x = self.flatten(batch[0])
        assert len(x) == len(snr)
        if is_simple:
            snr = timestep_embedding(snr, self.time_dim)
        else: 
            snr = timestep_embedding(snr, self.dim_dt)
            snr = self.time_mlp(snr)
        if len(batch) > 1:
            y = self.flatten(batch[1])
            assert len(y) == len(x)
            inputs = t.cat((x, y), 1)
        else:
            inputs = x
        
        h = []

        for resblock, downsample in self.downs:
            inputs = resblock(inputs, snr)
            h.append(inputs)
            inputs = downsample(inputs)

        inputs = self.mid_block(inputs, snr)

        for resblock, upsample in self.ups:
            inputs = t.cat((inputs, h.pop()), dim=1)
            inputs = resblock(inputs, snr)
            inputs = upsample(inputs)

        return self.final_layer(inputs)


class LinUnet(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y = 0,
        dim_mults = (2, 4, 8),
    ):
        super().__init__()

        # determine dimensions
        self.dim_dt = dim_x + dim_y
        self.time_dim = self.dim_dt * 4
        self.dim = self.dim_dt + self.time_dim
        dims = [self.dim, *map(lambda m: self.dim // m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers

        self.flatten = nn.Flatten()
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(
                nn.Linear(dim_in, dim_out)
            )

        mid_dim = dims[-1]
        self.mid_block = nn.Linear(mid_dim, mid_dim)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(
               nn.Linear(dim_out*2, dim_in)
            )

        self.final_layer = nn.Linear(self.dim, dim_x)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim_dt, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

    def forward(self, batch, snr, is_simple=False):
        x = self.flatten(batch[0])
        assert len(x) == len(snr)
        if is_simple:
            snr = timestep_embedding(snr, self.time_dim)
        else: 
            snr = timestep_embedding(snr, self.dim_dt)
            snr = self.time_mlp(snr)
        if len(batch) > 1:
            y = self.flatten(batch[1])
            assert len(y) == len(x)
            inputs = t.cat((x, y, snr), 1)
        else:
            inputs = t.cat((x, snr), 1)
        
        h = []
        for downsample in self.downs:
            inputs = downsample(inputs)
            h.append(inputs)

        inputs = self.mid_block(inputs)

        for upsample in self.ups:
            inputs = t.cat((inputs, h.pop()), dim=1)
            inputs = upsample(inputs)

        return self.final_layer(inputs)