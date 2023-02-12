import math
import torch as t
import numpy as np
from scipy.special import expit
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

def viz_mse_change(mses, log_eigs, logsnrs, d):
    """Visualization for multiple mse curves (IDDPM and DDPM)"""
    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

    length = len(mses) // 2

    fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(6, 5))

    base_logsnrs = t.linspace(logsnrs[0][0], logsnrs[0][-1], 100)
    base_baseline = np.array([d / (1. + np.exp(-logsnr)) for logsnr in base_logsnrs])
    base_baseline2 = t.sigmoid(base_logsnrs + log_eigs.view((-1, 1))).sum(axis=0).numpy()
    ax = axs[1]
    ax.plot(base_logsnrs, base_baseline, label='$N(0,1)$ MMSE')
    ax.plot(base_logsnrs, base_baseline2, lw=3, label='$N(\mu,\Sigma)$ MMSE')
    cnt = 0
    for y, x in zip(mses[:length], logsnrs[:length]):
        if cnt == 0:
            ax.plot(x, y, label=f'before fine-tuning') 
        else:
            ax.plot(x, y, color=cmap((cnt)/len(mses)), label=f'epoch{cnt}')
        cnt += 1

    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\gamma$)')
    ax.set_yticks([0, d/4, d/2, 3*d/4, d], ['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.set_title('DDPM')

    base_logsnrs = t.linspace(logsnrs[length-1][0], logsnrs[length-1][-1], 100)
    base_baseline = np.array([d / (1. + np.exp(-logsnr)) for logsnr in base_logsnrs])
    base_baseline2 = t.sigmoid(base_logsnrs + log_eigs.view((-1, 1))).sum(axis=0).numpy()
    ax = axs[0]
    ax.plot(base_logsnrs, base_baseline, label='$N(0,1)$ MMSE')
    ax.plot(base_logsnrs, base_baseline2, lw=3, label='$N(\mu,\Sigma)$ MMSE')
    cnt = 0
    for y, x in zip(mses[length:], logsnrs[length:]):
        if cnt == 0:
            ax.plot(x, y, label=f'before fine-tuning') 
        else:
            ax.plot(x, y, color=cmap(cnt/len(mses)), label=f'epoch{cnt}')
        cnt += 1

    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\gamma$)')
    ax.set_yticks([0, d/4, d/2, 3*d/4, d], ['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.legend(fontsize = 'x-small')
    ax.set_title('IDDPM')

    fig.set_tight_layout(True)

    return fig

def viz_loss_change(train_loss, val_loss, nll):
    """Visualization for multiple loss values (IDDPM and DDPM)"""
    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(6, 5))

    epochs = len(val_loss) // 2
    niter = len(train_loss) // 2

    ax = axs[1]
    ax.plot(np.arange(1, niter+1), train_loss[:niter], '--o', label='train')
    # ax.plot(np.arange(0, epochs), val_loss[:epochs], '-o', label='test_loss')
    ax.plot(np.arange(0, epochs), nll[:epochs], '-o', label='test')
    ax.legend()
    ax.set_ylabel('NLL (bpd)')
    ax.set_xlabel('Epochs')
    ax.set_title('DDPM')

    ax = axs[0]
    ax.plot(np.arange(1, niter+1), train_loss[niter:], '--o', label='train')
    # ax.plot(np.arange(0, epochs), val_loss[epochs:], '-o', label='test_loss')
    ax.plot(np.arange(0, epochs), nll[epochs:], '-o', label='test')
    ax.legend()
    ax.set_ylabel('NLL (bpd)')
    ax.set_xlabel('Epochs')
    ax.set_title('IDDPM')
    fig.set_tight_layout(True)

    return fig

def plot_mse_loss():
    mses = []
    logsnrs = []
    train_loss = []
    test_loss = []
    nll = []
    epoch = 10

    covariance = t.load('./covariance/cifar_covariance.pt')
    mu, U, log_eigs = covariance

    # load ddpm mse curves and loss
    train_loss1 = np.load(f'./results/fine_tune/ddpm/train_loss_all.npy', allow_pickle=True) / 32 / 32 / 3 / np.log(2.0)
    train_loss.extend(train_loss1.tolist()[:epoch])
    test_loss1 = np.load(f'./results/fine_tune/ddpm/test_loss_all.npy', allow_pickle=True) / 32 / 32 / 3 / np.log(2.0)
    test_loss.extend(test_loss1.tolist()[:epoch+1])
    for i in range(epoch+1):
        x = np.load(f'./results/fine_tune/ddpm/results_epoch{i}.npy', allow_pickle=True)
        mses.append(x.item()['mses'])
        logsnrs.append(x.item()['logsnr'])
        nll.append(x.item()['nll (bpd)'].cpu().numpy())

    # load iddpm mse curves and loss
    train_loss2 = np.load(f'./results/fine_tune/iddpm/train_loss_all.npy', allow_pickle=True) / 32 / 32 / 3 / np.log(
        2.0)
    train_loss.extend(train_loss2.tolist()[:epoch])
    test_loss2 = np.load(f'./results/fine_tune/iddpm/test_loss_all.npy', allow_pickle=True) / 32 / 32 / 3 / np.log(2.0)
    test_loss.extend(test_loss2.tolist()[:epoch+1])
    for i in range(epoch+1):
        x = np.load(f'./results/fine_tune/iddpm/results_epoch{i}.npy', allow_pickle=True)
        mses.append(x.item()['mses'])
        logsnrs.append(x.item()['logsnr'])
        nll.append(x.item()['nll (bpd)'].cpu().numpy())

    print("train loss: {} \ntest loss: {} \ntest nll:{}".format(train_loss, test_loss, nll))

    fig1 = viz_mse_change(mses, log_eigs, logsnrs, 32*32*3)
    fig2 = viz_loss_change(train_loss, test_loss, nll)
    fig1.savefig(f'./results/figs/MSE.png')
    fig2.savefig(f'./results/figs/LOSS.png')
    fig1.savefig(f'./results/figs/MSE.pdf')
    fig2.savefig(f'./results/figs/LOSS.pdf')

def plot_mse_comparison():
    ddpm = np.load('./results/fine_tune/ddpm/results_epoch0_1000.npy', allow_pickle=True)[0]
    iddpm = np.load('./results/fine_tune/iddpm/results_epoch0_1000.npy', allow_pickle=True)[0]
    ddpm_tune = np.load('./results/fine_tune/ddpm/results_epoch10_1000.npy', allow_pickle=True)[0]
    iddpm_tune = np.load('./results/fine_tune/iddpm/results_epoch10_1000.npy', allow_pickle=True)[0]

    # Properties of data used
    d = 32 * 32 * 3
    clip = 4
    log_eigs = t.load('./covariance/cifar_covariance.pt')[2]  # Load cached spectrum for speed
    mmse_g = ddpm['mmse_g']
    logsnr = ddpm['logsnr']  # With the same random seed on the same device, logsnrs are the same
    mmse_g_1 = d / (1 + np.exp(-logsnr))
    # Used to estimate good range for integration
    loc_logsnr = -log_eigs.mean().item()
    scale_logsnr = t.sqrt(1 + 3. / math.pi * log_eigs.var()).item()
    w = scale_logsnr * np.tanh(clip / 2) / (
                expit((logsnr - loc_logsnr) / scale_logsnr) * expit(-(logsnr - loc_logsnr) / scale_logsnr))

    min_mse = reduce(t.minimum, [mmse_g, iddpm_tune['mses'], iddpm['mses'], ddpm_tune['mses'], ddpm['mses']])

    min_mse_discrete = reduce(t.minimum, [mmse_g, ddpm['mses'], iddpm['mses'], iddpm_tune['mses'], ddpm_tune['mses'],
                                          ddpm['mses_round_xhat'], iddpm[
                                              'mses_round_xhat']])  # iddpm_tune_soft['mses_round_xhat'], ddpm_tune_soft['mses_round_xhat']

    cmap2 = sns.color_palette("Paired")
    cmap = sns.color_palette()

    # Continuous (Fig. 1)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 6, forward=True)
    ax.plot(logsnr, mmse_g_1, label='MMSE$_\epsilon$ for $N(0, I)$', color=cmap[1])
    ax.plot(logsnr, ddpm['mmse_g'], label='MMSE$_\epsilon$ for $N(\\mu, \\Sigma)$', color=cmap[8])
    ax.plot(logsnr, ddpm['mses'], label='DDPM', color=cmap2[0])
    ax.plot(logsnr, iddpm['mses'], label='IDDPM', color=cmap2[2])
    ax.plot(logsnr, ddpm_tune['mses'], label='DDPM-tuned', color=cmap2[1])
    ax.plot(logsnr, iddpm_tune['mses'], label='IDDPM-tuned', color=cmap2[3])
    ax.set_xlabel('$\\alpha$ (log SNR)')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon(z_\\alpha, \\alpha))^2]$')
    ax.fill_between(logsnr, min_mse, ddpm['mmse_g'], alpha=0.1)
    ax.set_yticks([0, d / 4, d / 2, 3 * d / 4, d])
    ax.set_yticklabels(['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.legend(loc='lower right')

    fig.set_tight_layout(True)
    fig.savefig('./results/figs/cont_density.pdf')

    # Discrete (Fig. 2)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 6, forward=True)
    ax.plot(logsnr, ddpm['mses'], label='DDPM', color=cmap2[0])
    ax.plot(logsnr, iddpm['mses'], label='IDDPM', color=cmap2[2])
    ax.plot(logsnr, ddpm_tune['mses'], label='DDPM-tuned', color=cmap2[1])
    ax.plot(logsnr, iddpm_tune['mses'], label='IDDPM-tuned', color=cmap2[3])
    ax.plot(logsnr, ddpm['mses_round_xhat'], label='round(DDPM)', color=cmap2[5])
    ax.plot(logsnr, iddpm['mses_round_xhat'], label='round(IDDPM)', color=cmap2[9])
    ax.plot(logsnr, ddpm_tune['mses_round_xhat'], '--', label='round(DDPM-tuned)', color=cmap2[4])
    ax.plot(logsnr, iddpm_tune['mses_round_xhat'], '--', label='round(IDDPM-tuned)', color=cmap2[8])

    ax.set_xlabel('$\\alpha$ (log SNR)')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon(z_\\alpha, \\alpha))^2]$')
    ax.fill_between(logsnr, min_mse_discrete, alpha=0.1)
    ax.set_yticks([0, d / 4, d / 2, 3 * d / 4, d])
    ax.set_yticklabels(['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.legend(loc='upper left')

    fig.set_tight_layout(True)
    fig.savefig('./results/figs/disc_density.pdf')

def print_tables():
    ddpm = np.load('./results/fine_tune/ddpm/results_epoch0_1000.npy', allow_pickle=True)[0]
    iddpm = np.load('./results/fine_tune/iddpm/results_epoch0_1000.npy', allow_pickle=True)[0]
    ddpm_tune = np.load('./results/fine_tune/ddpm/results_epoch10_1000.npy', allow_pickle=True)[0]
    iddpm_tune = np.load('./results/fine_tune/iddpm/results_epoch10_1000.npy', allow_pickle=True)[0]

    # Properties of data used
    d = 32 * 32 * 3
    clip = 4
    log_eigs = t.load('./covariance/cifar_covariance.pt')[2]  # Load cached spectrum for speed
    h_g = 0.5 * d * math.log(2 * math.pi * math.e) + 0.5 * log_eigs.sum().item()
    mmse_g = ddpm['mmse_g']
    logsnr = ddpm['logsnr']  # With the same random seed on the same device, logsnrs are the same
    # Used to estimate good range for integration
    loc_logsnr = -log_eigs.mean().item()
    scale_logsnr = t.sqrt(1 + 3. / math.pi * log_eigs.var()).item()
    w = scale_logsnr * np.tanh(clip / 2) / (
                expit((logsnr - loc_logsnr) / scale_logsnr) * expit(-(logsnr - loc_logsnr) / scale_logsnr))
    # select 100 logsnrs out of 1k
    seed = [1024, 123, 7, 10, 11, 14, 15, 18, 19, 30]

    ddpm_nll_bpd = []
    iddpm_nll_bpd = []
    ddpm_tune_nll_bpd = []
    iddpm_tune_nll_bpd = []
    ddpm_nll_deq_bpd = []
    iddpm_nll_deq_bpd = []
    ddpm_tune_nll_deq_bpd = []
    iddpm_tune_nll_deq_bpd = []
    ddpm_nll_bpd_disc = []
    iddpm_nll_bpd_disc = []
    ddpm_tune_nll_bpd_disc = []
    iddpm_tune_nll_bpd_disc = []
    nll_bpd = []
    nll_bpd_discrete = []
    ens_deq = []

    def cont_bpd_from_mse(w, mses, mmse_g):
        nll_nats = h_g - 0.5 * (w * t.clamp(mmse_g.cpu() - mses.cpu(), 0.)).mean()
        return nll_nats / math.log(2) / d

    def disc_bpd_from_mse(w, mses):
        return 0.5 * (w * mses.cpu()).mean() / math.log(2) / d

    min_mse_1000 = reduce(t.minimum, [mmse_g, iddpm_tune['mses'], iddpm['mses'], ddpm_tune['mses'], ddpm['mses']])
    nll_bpd_1000 = cont_bpd_from_mse(w, min_mse_1000, mmse_g)
    min_mse_discrete_1000 = reduce(t.minimum, [mmse_g, ddpm['mses'], iddpm['mses'], iddpm_tune['mses'],
                                          ddpm_tune['mses'], ddpm['mses_round_xhat'],
                                          iddpm['mses_round_xhat']])
    nll_bpd_discrete_1000 = disc_bpd_from_mse(w, min_mse_discrete_1000)
    min_deq_1000 = reduce(t.minimum,
                     [mmse_g, ddpm['mses_dequantize'], iddpm['mses_dequantize'], ddpm_tune['mses_dequantize'],
                      iddpm_tune['mses_dequantize']])
    ens_deq_1000 = cont_bpd_from_mse(w, min_deq_1000, mmse_g) + np.log(127.5) / np.log(2)

    for sd in seed:
        t.manual_seed(sd)
        idx, _ = t.randint(0, 1000, (100,)).sort()
        mmse_g_idx = ddpm['mmse_g'][idx]
        logsnr_idx = ddpm['logsnr'][idx]  # With the same random seed on the same device, logsnrs are the same
        # Used to estimate good range for integration
        loc_logsnr = -log_eigs.mean().item()
        scale_logsnr = t.sqrt(1 + 3. / math.pi * log_eigs.var()).item()
        w_idx = scale_logsnr * np.tanh(clip / 2) / (
                    expit((logsnr_idx - loc_logsnr) / scale_logsnr) * expit(-(logsnr_idx - loc_logsnr) / scale_logsnr))

        min_mse = reduce(t.minimum, [mmse_g_idx, iddpm_tune['mses'][idx], iddpm['mses'][idx], ddpm_tune['mses'][idx],
                                     ddpm['mses'][idx]])
        nll_bpd.append(cont_bpd_from_mse(w_idx, min_mse, mmse_g_idx))

        min_mse_discrete = reduce(t.minimum, [mmse_g_idx, ddpm['mses'][idx], iddpm['mses'][idx], iddpm_tune['mses'][idx],
                                              ddpm_tune['mses'][idx], ddpm['mses_round_xhat'][idx],
                                              iddpm['mses_round_xhat'][idx]])
        nll_bpd_discrete.append(disc_bpd_from_mse(w_idx, min_mse_discrete))

        min_deq = reduce(t.minimum, [mmse_g_idx, ddpm['mses_dequantize'][idx], iddpm['mses_dequantize'][idx],
                                     ddpm_tune['mses_dequantize'][idx], iddpm_tune['mses_dequantize'][idx]])
        ens_deq.append(cont_bpd_from_mse(w_idx, min_deq, mmse_g_idx) + np.log(127.5) / np.log(2))

        ddpm_nll_bpd.append(cont_bpd_from_mse(w_idx, ddpm['mses'][idx], mmse_g_idx))
        iddpm_nll_bpd.append(cont_bpd_from_mse(w_idx, iddpm['mses'][idx], mmse_g_idx))
        ddpm_tune_nll_bpd.append(cont_bpd_from_mse(w_idx, ddpm_tune['mses'][idx], mmse_g_idx))
        iddpm_tune_nll_bpd.append(cont_bpd_from_mse(w_idx, iddpm_tune['mses'][idx], mmse_g_idx))

        ddpm_nll_deq_bpd.append(cont_bpd_from_mse(w_idx, ddpm['mses_dequantize'][idx], mmse_g_idx) + np.log(127.5) / np.log(2))
        iddpm_nll_deq_bpd.append(cont_bpd_from_mse(w_idx, iddpm['mses_dequantize'][idx], mmse_g_idx) + np.log(127.5) / np.log(2))
        ddpm_tune_nll_deq_bpd.append(
            cont_bpd_from_mse(w_idx, ddpm_tune['mses_dequantize'][idx], mmse_g_idx) + np.log(127.5) / np.log(2))
        iddpm_tune_nll_deq_bpd.append(
            cont_bpd_from_mse(w_idx, iddpm_tune['mses_dequantize'][idx], mmse_g_idx) + np.log(127.5) / np.log(2))

        ddpm_nll_bpd_disc.append(disc_bpd_from_mse(w_idx, ddpm['mses_round_xhat'][idx]))
        iddpm_nll_bpd_disc.append(disc_bpd_from_mse(w_idx, iddpm['mses_round_xhat'][idx]))
        ddpm_tune_nll_bpd_disc.append(disc_bpd_from_mse(w_idx, ddpm_tune['mses_round_xhat'][idx]))
        iddpm_tune_nll_bpd_disc.append(disc_bpd_from_mse(w_idx, iddpm_tune['mses_round_xhat'][idx]))

    print('Continuous (Table 1)')
    print('DDPM &  &  & {:.2f} \\\\'.format(np.mean(ddpm_nll_bpd)))
    print('IDDPM &  &  & {:.2f} \\\\'.format(np.mean(iddpm_nll_bpd)))
    print('\\midrule')
    print('DDPM(tune)&  &  & {:.2f} \\\\'.format(np.mean(ddpm_tune_nll_bpd)))
    print('IDDPM(tune) &  &  & {:.2f} \\\\'.format(np.mean(iddpm_tune_nll_bpd)))
    print('\\midrule')
    print('Ensemble &  &  & {:.2f} \\\\'.format(np.mean(nll_bpd)))

    print('\n\n\nDiscrete (Table 2)')
    print('DDPM &  & {:.2f} & {:.2f} \\\\'.format(disc_bpd_from_mse(w, ddpm['mses_round_xhat']),
                                                  np.mean(ddpm_nll_deq_bpd)))
    print('IDDPM &  & {:.2f} & {:.2f} \\\\'.format(disc_bpd_from_mse(w, iddpm['mses_round_xhat']),
                                                   np.mean(iddpm_nll_deq_bpd)))
    print('\\midrule')
    print('DDPM(tune)&  & {:.2f} & {:.2f} \\\\'.format(disc_bpd_from_mse(w, ddpm_tune['mses_round_xhat']),
                                                       np.mean(ddpm_tune_nll_deq_bpd)))
    print('IDDPM(tune) &  & {:.2f} & {:.2f} \\\\'.format(disc_bpd_from_mse(w, iddpm_tune['mses_round_xhat']),
                                                         np.mean(iddpm_tune_nll_deq_bpd)))
    print('\\midrule')
    print('Ensemble &  &  {:.2f} & {:.2f} \\\\'.format(nll_bpd_discrete_1000, np.mean(ens_deq)))

    # variants
    print('\n\n\nStandard Deviation (Table 3)')
    print('DDPM - nll (bpd) std: {:.5f}, nll-discrete (bpd) std: {:.5f}, nll (bpd) - dequantize std: {:.5f}'.format(
        ddpm['nll (bpd) - std'], ddpm['nll-discrete (bpd) - std'], ddpm['nll (bpd) - dequantize - std']))
    print(
        'IDDPM - nll (bpd) std: {:.5f}, nll-discrete (bpd) std: {:.5f}, nll (bpd) - dequantize std: {:.5f}'.format(
            iddpm['nll (bpd) - std'], iddpm['nll-discrete (bpd) - std'], iddpm['nll (bpd) - dequantize - std']))
    print(
        'DDPM-tune - nll (bpd) std: {:.5f}, nll-discrete (bpd) std: {:.5f}, nll (bpd) - dequantize std: {:.5f}'.format(
            ddpm_tune['nll (bpd) - std'], ddpm_tune['nll-discrete (bpd) - std'],
            ddpm_tune['nll (bpd) - dequantize - std']))
    print(
        'IDDPM-tune - nll (bpd) std: {:.5f}, nll-discrete (bpd) std: {:.5f}, nll (bpd) - dequantize std: {:.5f}'.format(
            iddpm_tune['nll (bpd) - std'], iddpm_tune['nll-discrete (bpd) - std'],
            iddpm_tune['nll (bpd) - dequantize - std']))

    print('\n\n\nContinuous (Table 4)')
    print('DDPM & {:.2f} & {:.2f}|{:.5f} \\\\'.format(ddpm['nll (bpd)'], np.mean(ddpm_nll_bpd), np.std(ddpm_nll_bpd)))
    print('IDDPM & {:.2f} & {:.2f}|{:.5f} \\\\'.format(iddpm['nll (bpd)'], np.mean(iddpm_nll_bpd), np.std(iddpm_nll_bpd)))
    print('\\midrule')
    print('DDPM(tune) & {:.2f}|{:.5f} \\\\'.format(ddpm_tune['nll (bpd)'], np.mean(ddpm_tune_nll_bpd), np.std(ddpm_tune_nll_bpd)))
    print('IDDPM(tune) & {:.2f}|{:.5f} \\\\'.format(iddpm_tune['nll (bpd)'], np.mean(iddpm_tune_nll_bpd), np.std(iddpm_tune_nll_bpd)))
    print('\\midrule')
    print('Ensemble & {:.2f}|{:.5f} \\\\'.format(nll_bpd_1000, np.mean(nll_bpd), np.std(nll_bpd)))

    print('\n\n\nDiscrete (Table 5)')
    print('DDPM & {:.2f} & {:.2f}|{:.5f} \\\\'.format(disc_bpd_from_mse(w, ddpm['mses_round_xhat']), np.mean(ddpm_nll_bpd_disc), np.std(ddpm_nll_bpd_disc)))
    print('IDDPM & {:.2f} & {:.2f}|{:.5f} \\\\'.format(disc_bpd_from_mse(w, iddpm['mses_round_xhat']), np.mean(iddpm_nll_bpd_disc), np.std(iddpm_nll_bpd_disc)))
    print('\\midrule')
    print('DDPM(tune) & {:.2f} & {:.2f}|{:.5f} \\\\'.format(disc_bpd_from_mse(w, ddpm_tune['mses_round_xhat']), np.mean(ddpm_tune_nll_bpd_disc), np.std(ddpm_tune_nll_bpd_disc)))
    print('IDDPM(tune) & {:.2f} & {:.2f}|{:.5f} \\\\'.format(disc_bpd_from_mse(w, iddpm_tune['mses_round_xhat']), np.mean(iddpm_tune_nll_bpd_disc), np.std(iddpm_tune_nll_bpd_disc)))
    print('\\midrule')
    print('Ensemble & {:.2f} & {:.2f}|{:.5f} \\\\'.format(nll_bpd_discrete_1000, np.mean(nll_bpd_discrete), np.std(nll_bpd_discrete)))

    print('\n\n\nUniform Dequantization (Table 6)')
    print('DDPM & {:.2f} & {:.2f}|{:.5f} \\\\'.format(ddpm['nll-discrete-limit (bpd) - dequantize'], np.mean(ddpm_nll_deq_bpd), np.std(ddpm_nll_deq_bpd)))
    print('IDDPM & {:.2f} & {:.2f}|{:.5f} \\\\'.format(iddpm['nll-discrete-limit (bpd) - dequantize'], np.mean(iddpm_nll_deq_bpd), np.std(iddpm_nll_deq_bpd)))
    print('\\midrule')
    print('DDPM(tune) & {:.2f} & {:.2f}|{:.5f} \\\\'.format(ddpm_tune['nll-discrete-limit (bpd) - dequantize'], np.mean(ddpm_tune_nll_deq_bpd), np.std(ddpm_tune_nll_deq_bpd)))
    print('IDDPM(tune) & {:.2f} & {:.2f}|{:.5f} \\\\'.format(iddpm_tune['nll-discrete-limit (bpd) - dequantize'], np.mean(iddpm_tune_nll_deq_bpd), np.std(iddpm_tune_nll_deq_bpd)))
    print('\\midrule')
    print('Ensemble & {:.2f} & {:.2f}|{:.5f} \\\\'.format(ens_deq_1000, np.mean(ens_deq), np.std(ens_deq)))

if __name__ == "__main__":
    plot_mse_comparison()  # plot Fig. 1 & 2
    plot_mse_loss() # plot Fig. 5 & 6 in section B.2
    print_tables() # print ALL Tables
    print("\n\n\nFor benchmark results, please read the README.md file in './benchmark/improved-diffusion'.")