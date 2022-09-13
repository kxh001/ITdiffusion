# Setup, initialization
import math
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.datasets import make_spd_matrix
import torch as t
import matplotlib.pyplot as plt
import os

# Internal imports
from nets import SimpleResNet
from utils import CustomDataset, viz
from diffusionmodel import DiffusionModel

device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train_resnet_gaussian(n_samples, n_features):
    """
    :para n_samples: # of gaussian data points
    :pare n_features: dimensionality of a data point
    """
    # Generate Gaussian data with known covariance
    prng = np.random.RandomState(1)
    cov = make_spd_matrix(n_features, random_state=prng)  # Generate a symmetric definite positive matrix.
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]

    rv = multivariate_normal(np.zeros(n_features), cov)
    x = rv.rvs(size=n_samples, random_state=1234)  # samples from this multivariate normal
    nll = - rv.logpdf(x)  # negative log likelihood for these samples
    x = t.from_numpy(x).type(t.FloatTensor)  # put in pytorch float tensor

    # plot the covariances
    vmax = cov.max()
    plt.imshow(cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)
    plt.axis('off')
    plt.title("True covariance")
    plt.savefig('figures/gauss_covariance.png')
    plt.close('all')

    # Make dataset with 90-10 train / validation split
    x = x.to(device)  # Small enough to go on GPU
    x_train, x_val = x[:(n_samples * 9) // 10], x[(n_samples * 9) // 10:]
    dataset = CustomDataset(x_train)
    val_dataset = CustomDataset(x_val)

    resnet = SimpleResNet(dim_x=n_features, dim_y=0) 
    resnet.to(device)
    dm = DiffusionModel(resnet)
    dm.fit(dataset, val_dataset, epochs=200, lr=1e-3, batch_size=32, use_optimizer='adam', verbose=False)
    fig = viz(dm.logs, d=n_features)
    out_path = os.path.join(r"D:\Research Result\P3_ICLR", f"viz.png")
    fig.savefig(out_path)

    mse, logsnr = dm.logs['mse_curves'][-1], dm.logs['logsnr_grid']
    mse = mse.cpu().numpy()
    logsnr = logsnr.cpu().numpy()
    snr = np.exp(logsnr)
    eps_mse = mse * snr

    mmse_g = n_features / (1 + snr)
    c = 0.5 * n_features * np.log(2 * math.pi *math.e)
    print("max entropy (entropy of N(0,I)): ", c)
    # Now integrate this
    print("h(x) = d/2 log 2 pi e - 1/2 int (mmse_g - mmse).\n"
          "Estimated with optimal estimator, x trapz integral: {:.3f}\n"
          "Estimated with optimal estimator, eps trapz integral: {:.3f}\n"
          "Estimated with MCI: {:.3f}\n".format(
        c - np.trapz(0.5 * (mmse_g - mse), x=snr),
        c - np.trapz(0.5 * (mmse_g - eps_mse), x=logsnr),
        dm.logs['val loss'][-1]
        ))
    print('\nGaussian Resnet experiments complete\n\n')

if __name__ == "__main__":
    train_resnet_gaussian(1000, 32)