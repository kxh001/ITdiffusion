# Setup, initialization
import math
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.datasets import make_spd_matrix
import torch as t
import matplotlib.pyplot as plt

# Internal imports
from nets import LinNet, SimpleResNet, GaussTrueMMSE
from utils import CustomDataset, viz, trunc_normal_integrate, logistic_integrate
from diffusionmodel import DiffusionModel

device = "cuda" if t.cuda.is_available() else "cpu"
print(f"Using {device} device")


def test_gaussian(n_samples=10000, n_features=32):

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

    # A ground truth optimal estimator for Gaussian data
    gt_net = GaussTrueMMSE(cov, device)
    gt_net.to(device)
    dm = DiffusionModel(gt_net)
    # there are no params to fit, but we run it once just to populate validation stats in train loop
    dm.fit(dataset, val_dataset, epochs=1, lr=1e-2, batch_size=500, verbose=False)
    fig = viz(dm.logs, d=n_features)
    fig.savefig('figures/gauss_test_curves.png')

    # Now lets compare the estimated MSE with Ground Truth (GT) optimal estimator, to the GT MMSE curve
    fig, axs = plt.subplots(1, 2)
    mse, logsnr = dm.logs['mse_curves'][0], dm.logs['logsnr_grid']
    snr = t.exp(logsnr)
    axs[0].plot(t.log(snr).cpu().numpy(), mse.cpu().numpy(), label='estimated MSE with GT x_hat estimator')

    gt_mmse = gt_net.mmse(snr.to(device)).cpu().numpy()  # Ground truth MMSE for this covariance
    axs[0].plot(t.log(snr).cpu().numpy(), gt_mmse, label='GT MMSE(snr)')
    axs[0].set_title('MMSE curves')
    axs[0].set_ylabel('MMSE Gap $(x)$')
    axs[0].set_xlabel('log SNR ($\gamma$)')
    fig.legend(loc='upper center')

    # Same plot - but gap with MMSE_Gaussian
    mmse_g = gt_net.d / (1 + snr.cpu().numpy())  # MMSE for N(0,I)
    axs[1].plot(t.log(snr).cpu().numpy(), mmse_g - mse.cpu().numpy(), label='estimated MSE with GT x_hat estimator')

    gt_mmse = gt_net.mmse(snr.to(device)).cpu().numpy()
    axs[1].plot(t.log(snr).cpu().numpy(), mmse_g - gt_mmse, label='GT MMSE(snr)')
    axs[1].set_title('Gap in MMSE')
    axs[1].set_ylabel('MMSE Gap $(x)$')
    axs[1].set_xlabel('log SNR ($\gamma$)')
    fig.savefig('figures/gauss_mmse_gt_versus_estimator.png')

#    N = 200000
#    logsnr = t.linspace(-10, 10, N)
    #gt_mmse = gt_net.mmse(t.exp(logsnr).to(device)).cpu()
    #mmse_g = gt_net.d / (1 + t.exp(logsnr)).cpu()  # MMSE for N(0,I)

    snr = snr.cpu().numpy()
    logsnr = np.log(snr)
    eps_mmse = gt_mmse * snr
    eps_mse = mse * snr

    # Gradient tests: failed due to inaccuracy or high variance
    # print("\n Test gradient of MMSE estimator")
    # x = x_val[0]
    # test_snr = t.tensor(1.4)
    # print('x', x, 'snr', test_snr)
    # print('true grad 1/2 mmse', gt_net.true_grad_mmse(x, test_snr))
    # print('estimate', dm.grad_mmse(x, test_snr, npoints=500))
    # print('estimate 0', dm.grad_mmse_0(x, test_snr))
    # print('estimate 1', dm.grad_mmse_1(x, t.log(t.tensor([test_snr, test_snr], device=x.device))))
    # print('from dm method', dm.grad_mmse_g(x,t.log(t.tensor([test_snr, test_snr], device=x.device))))

#     print("\n Test gradient")
#     x = x_val[0]
#     print('x', x)
#     print('true grad', gt_net.true_grad(x))
#     # print('estimate', dm.grad(x))
#     limit_snr = t.tensor([1000.], device=x.device)
#     print("use snr", limit_snr)
#     print('limit estimator', t.sqrt(1+limit_snr) * dm.model([x.view((1,-1))], limit_snr))
# #     print('model out', dm.model([x.view((1,-1))], limit_snr))
#     print('from dm method analytic grad', dm.grad_g(x))

    # Test pointwise MMSE
    fig = mmse_x_curve(dm, x[0])
    fig.savefig('figures/test_mmse_x_in_dist.png')
    fig = mmse_x_curve(dm, t.randn_like(x[0]))
    fig.savefig('figures/test_mmse_x_out_dist.png')
    fig = av_mmse_curve(dm, x_val[:100])
    fig.savefig('figures/test_av_mmse_in_dist.png')
    fig = av_mmse_curve(dm, t.randn_like(x_val[:100]))
    fig.savefig('figures/test_av_mmse_out_dist.png')

    # Numerically integrate to get h(x) using Eq. 8 from write-up, compare with ground truth.
    mci = dm.nll([x_val], logsnr_samples_per_x=10).mean().detach().cpu().numpy()
    mci1 = np.mean([dm.nll([x_val[i:i+1]], logsnr_samples_per_x=10).mean().detach().cpu().numpy() for i in range(len(x_val))])

    print("Compare Monte Carlo int estimate from full batch: "
          "{:.3f}\n and same estimate with batch size one: {:.3f}".format(mci, mci1))

    mcis = [dm.nll([x_val], logsnr_samples_per_x=60).mean().detach().cpu().numpy() for _ in range(20)]
    mci_mean, mci_std = np.mean(mcis), np.std(mcis)

    # import IPython; IPython.embed()

    c = 0.5 * gt_net.d * np.log(2 * math.pi *math.e)
    print("max entropy (entropy of N(0,I)): ", c)
    # Now integrate this
    print("h(x) = d/2 log 2 pi e - 1/2 int (mmse_g - mme).\n"
          "GT h(x): {:.3f}\n"
          "Integrate ground truth MMSE: {:.3f}\n"
          "Estimated with optimal estimator, x trapz integral: {:.3f}\n"
          "Estimated with optimal estimator, eps trapz integral: {:.3f}\n"
          "difference estimator: {:.3f}\n"
          "Monte Carlo int: {:.3f} +- {:.3f}\n" .format(
        gt_net.entropy(),
        c - np.trapz(0.5 * (mmse_g - gt_mmse), x=snr),
        c - np.trapz(0.5 * (mmse_g - mse.cpu().numpy()), x=snr),
        c - np.trapz(0.5 * (mmse_g - eps_mse.cpu().numpy()), x=logsnr),
        c - (0.5 * np.diff(eps_mmse) * 0.5 * (logsnr[:-1] +logsnr[1:])).sum(),
        mci_mean,
        mci_std))

    print('\nTest test_nll code.')
    val_dl = t.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    results = dm.test_nll(val_dl, npoints=100)
    print(results)
    print('\nGaussian experiments complete\n\n')



def test_tn_integrate():
    """Test numerically integrating some functions with known integral, f(a)"""
    print('\nIntegration test\n')

    a, w = trunc_normal_integrate(1000, 0., 1., clip=4, device=device)
    rv = t.distributions.Normal(0., 1.)
    integrand = t.exp(rv.log_prob(a)) * w
    print("integrate N(0,1) ~ 1 w/ perfect weights. {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))

    a, w = trunc_normal_integrate(1000, 1., 3., clip=4, device=device)
    rv = t.distributions.Normal(1., 3.)
    integrand = t.exp(rv.log_prob(a)) * w
    print("integrate N(1,3) ~ 1 w/ perfect weights. {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))

    a, w = trunc_normal_integrate(1000, 0., 2., clip=4, device=device)
    rv = t.distributions.Normal(0., 1.)
    integrand = t.exp(rv.log_prob(a)) * w
    print("integrate N(0,1) ~ 1 w/ too much weight on tails? {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))

    a, w = trunc_normal_integrate(1000, 0., 0.5, clip=3, device=device)
    integrand = t.exp(rv.log_prob(a)) * w
    print("integrate N(0,1) ~ 1 w/ not enough weight on tails? {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))


    print('\nIntegration test complete\n')


def test_logistic_integrate(deterministic=False):
    """Test numerically integrating some functions with known integral, f(a)"""
    print('\nIntegration test with logistic integrator\n')

    a, w = logistic_integrate(1000, 0., 1., device=device, deterministic=deterministic)
    rv = t.distributions.Normal(0., 1.)
    integrand = t.exp(rv.log_prob(a)) * w
    print("integrate N(0,1) ~ 1 w/ correct loc/scale. {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))

    a, w = logistic_integrate(1000, 1., 3., device=device, deterministic=deterministic)
    rv = t.distributions.Normal(1., 3.)
    integrand = t.exp(rv.log_prob(a)) * w
    print("integrate N(1,3) ~ 1 w/ correct loc/scale. {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))

    a, w = logistic_integrate(1000, 0., 2., device=device, deterministic=deterministic)
    rv = t.distributions.Normal(0., 1.)
    integrand = t.exp(rv.log_prob(a)) * w
    print("integrate N(0,1) ~ 1 w/ too much weight on tails? {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))

    a, w = logistic_integrate(1000, 0., 0.5, device=device, deterministic=deterministic)
    integrand = t.exp(rv.log_prob(a)) * w
    print("integrate N(0,1) ~ 1 w/ not enough weight on tails? {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))
    # Less weight on tails is actually better, in this case, because logistic has heavier tails than Gaussian

    a, w = logistic_integrate(1000, 0., 1., device=device, deterministic=deterministic)
    integrand = t.exp(-a) / t.square(1+t.exp(-a)) * w  # logistic distribution, should integrate to one with low variance
    print("integrate logistic ~ 1 w/ correct loc/scale. {:.3f} +- {:.3f} / sqrt(n)".format(integrand.mean(), integrand.std()))
    # Result is less than 1, because we clip the tails by default to avoid large weights

    print('\nIntegration test complete\n')


def mmse_x_curve(dm, x):
    """Plot estimates of MMSE_G(x, logsnr) and return fig."""
    logsnrs = t.linspace(-10, 10, 100, device=x.device)
    mmse_x = dm.mmse_g_x(x, logsnrs)
    mses_x = dm.mse([x.repeat(len(logsnrs), 1)], logsnrs, mse_type='x')
    mmse = dm.mmse_g(logsnrs) * t.exp(-logsnrs)
    mses_av = 0
    for i in range(40):
        mses_av += dm.mse([x.repeat(len(logsnrs), 1)], logsnrs, mse_type='x') / 40.
    fig, ax = plt.subplots(1)
    ax.scatter(logsnrs.cpu(), mmse_x.detach().cpu(), label='gt')
    ax.scatter(logsnrs.cpu(), mses_av.detach().cpu(), label='est avg')
    ax.scatter(logsnrs.cpu(), mses_x.detach().cpu(), label='est with opt')
    ax.scatter(logsnrs.cpu(), mmse.detach().cpu(), label='mmse(logsnr)')
    ax.set_xlabel('log SNR')
    ax.set_ylabel('$mmse_G(x, log SNR)$')
    fig.legend(loc='upper right')
    return fig

def av_mmse_curve(dm, xs):
    """Look at many mmse(x,snr) curves and see that the average is correct"""
    with t.no_grad():
        logsnrs = t.linspace(-10, 10, 100, device=xs[0].device)
        fig, ax = plt.subplots(1)
        ax.set_xlabel('log SNR')
        ax.set_ylabel('$mmse_G(x, log SNR)$')
        mmse = dm.mmse_g(logsnrs) * t.exp(-logsnrs)
        ax.scatter(logsnrs.cpu(), mmse.detach().cpu(), label='mmse(logsnr)')

        mmse_x = []
        for i, x in enumerate(xs):
            mmse_x.append(dm.mmse_g_x(x, logsnrs).squeeze(0))
            ax.plot(logsnrs.cpu(), mmse_x[-1].detach().cpu(), alpha=0.1)
        mmse_av = t.stack(mmse_x).mean(dim=0)
    ax.scatter(logsnrs.cpu(), mmse_av.detach().cpu(), label='E[mmse(x, log SNR)]')
    fig.legend(loc='upper right')
    return fig


def test_snr_detect(n_samples=1000, n_features=32):

    # Generate Gaussian data with known covariance
    prng = np.random.RandomState(1)
    cov = make_spd_matrix(n_features, random_state=prng)  # Generate a symmetric definite positive matrix.
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]

    rv = multivariate_normal(np.zeros(n_features), cov)
    x = rv.rvs(size=n_samples, random_state=1234)  # samples from this multivariate normal
    x = t.from_numpy(x).type(t.FloatTensor)  # put in pytorch float tensor

    # Make dataset with 90-10 train / validation split
    x = x.to(device)  # Small enough to go on GPU
    dataset = CustomDataset(x)

    # A ground truth optimal estimator for Gaussian data
    gt_net = GaussTrueMMSE(cov, device)
    gt_net.to(device)
    dm = DiffusionModel(gt_net)
    # there are no params to fit, but we run it once just to populate validation stats in train loop
    dm.fit(dataset, val_dataset=dataset, epochs=1, lr=1e-2, batch_size=500, verbose=False)
    fig = viz(dm.logs, d=n_features)
    fig.savefig('figures/detect_snr_test_mse_curve.png')

    # Now lets generate data, z, from
    # logsnr0 = t.ones_like(x[:, 0]) * (- 2.5)
    # logsnr_del = t.ones_like(x[:, 0]) * 2.  # Just a large SNR - adds little noise
    # with t.no_grad():
    #     z0, eps0 = dm.noisy_channel(x, logsnr0)
    #     z1, eps1 = dm.noisy_channel(z0, logsnr_del)
    #     logsnr1 = logsnr0 - t.log1p(t.exp(-logsnr_del) + t.exp(logsnr0-logsnr_del))  # The implied total snr
    #
    #     E_x_given_z1 = dm.x_hat(z1, logsnr1)
    #     c1 = t.sqrt(1 + t.exp(-logsnr_del))
    #     c0 = t.exp(-0.5 * logsnr_del) * t.sqrt((1 + t.exp(logsnr0)) / (1 + t.exp(logsnr0) + t.exp(logsnr_del)))
    #     # c0 = t.sqrt(t.exp(logsnr0) * (1 + t.exp(logsnr0))) / (1. + t.exp(logsnr0) + t.exp(logsnr1))
    #     # c1 = t.sqrt(t.exp(logsnr1) * (1 + t.exp(logsnr1))) / (1. + t.exp(logsnr0) + t.exp(logsnr1))
    #     E_z0_given_z1 = c1[0] * z1 - c0[0] * dm.model([z1], t.exp(logsnr1))
    #     err = t.square(z0 - E_z0_given_z1).flatten(start_dim=1).sum(dim=1).mean(dim=0)
    #     err0 = t.square(z0 - E_x_given_z1).flatten(start_dim=1).sum(dim=1).mean(dim=0)
    #     err_z = t.square(z0 - z1).flatten(start_dim=1).sum(dim=1).mean(dim=0)
    #
    #     E_z0_given_z1 = dm.e_z0_given_z1(z1, t.exp(logsnr1), t.exp(logsnr_del))
    #     err_check = t.square(z0 - E_z0_given_z1).flatten(start_dim=1).sum(dim=1).mean(dim=0)
    #     print(err, err_check, err0, err_z, 'want first to be smallest, first two match')
    #
    #     errs = []
    #     steins = []
    #     with t.no_grad():
    #         for logsnr in t.linspace(-5, 15, 40):
    #             logsnr1 = logsnr - t.log1p(t.exp(-logsnr_del) + t.exp(logsnr-logsnr_del))  # The implied total snr
    #             E_z0_given_z1 = dm.e_z0_given_z1(z1, t.exp(logsnr1), t.exp(logsnr_del))
    #             this_err = t.square(z0 - E_z0_given_z1).flatten(start_dim=1).sum(dim=1).mean(dim=0).cpu().item()
    #
    #             stein = (z0 * dm.model([z0], t.exp(logsnr * t.ones_like(z0[:,0])))).flatten(start_dim=1).sum(dim=1)
    #             steins.append((logsnr, stein.cpu().numpy()))
    #             errs.append((logsnr, this_err))
    #             print(errs[-1])
    #
    #     fig, ax = plt.subplots(1)
    #     ax.scatter(*zip(*errs))
    #     ax.axvline(logsnr0[0].cpu().item())
    #     ax.set_xlabel('log SNR')
    #     ax.set_ylabel('$E[(z_0 - E(z_0|z_1))^2]$')
    #     fig.savefig('figures/test_detect_snr.png')
    #
    #     # Naive Stein critic is not very good at figuring out correct log SNR.
    #     fig2, ax2 = plt.subplots(1)
    #     logsnrs = list(zip(*steins))[0]
    #     stein_av = [np.mean(s) for _, s in steins]
    #     ax2.scatter(logsnrs, stein_av)
    #     cross = [32. / np.sqrt(1+np.exp(l)) for l in logsnrs]
    #     plt.scatter(logsnrs, cross)
    #     ax2.set_xlabel('log SNR')
    #     ax2.set_title('Crossing point is detected log SNR')


    fig, axs = plt.subplots(1, 2)
    mse, logsnr = dm.logs['mse_curves'][0], dm.logs['logsnr_grid']
    snr = t.exp(logsnr)
    axs[0].plot(t.log(snr).cpu().numpy(), mse.cpu().numpy(), label='estimated MSE with GT x_hat estimator')

    gt_mmse = gt_net.mmse(snr.to(device)).cpu().numpy()  # Ground truth MMSE for this covariance
    axs[0].plot(t.log(snr).cpu().numpy(), gt_mmse, label='GT MMSE(snr)')
    axs[0].set_title('MMSE curves')
    axs[0].set_ylabel('MMSE Gap $(x)$')
    axs[0].set_xlabel('log SNR ($\gamma$)')
    fig.legend(loc='upper center')

    # Same plot - but gap with MMSE_Gaussian
    mmse_g = gt_net.d / (1 + snr.cpu().numpy())  # MMSE for N(0,I)
    axs[1].plot(t.log(snr).cpu().numpy(), mmse_g - mse.cpu().numpy(), label='estimated MSE with GT x_hat estimator')

    gt_mmse = gt_net.mmse(snr.to(device)).cpu().numpy()
    axs[1].plot(t.log(snr).cpu().numpy(), mmse_g - gt_mmse, label='GT MMSE(snr)')
    axs[1].set_title('Gap in MMSE')
    axs[1].set_ylabel('MMSE Gap $(x)$')
    axs[1].set_xlabel('log SNR ($\gamma$)')
    fig.savefig('figures/gauss_mmse_gt_versus_estimator.png')


if __name__ == "__main__":
    test_snr_detect()
    test_gaussian()  # Batch one test is slow, but worth checking if integrator is changed
    test_tn_integrate()
    test_logistic_integrate()
    print("REDO logistic integrate test, deterministic=TRUE")
    test_logistic_integrate(deterministic=True)