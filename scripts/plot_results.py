import torch as t
import matplotlib.pyplot as plt
from scipy.special import expit
from matplotlib import cm
import seaborn as sns
import numpy as np
import sys
import math
from utils import logistic_integrate
from functools import reduce
plt.style.use('seaborn-paper')
import seaborn as sns
sys.path.append('..')
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})

def viz_one(mses, log_eigs, logsnrs, d=2):
    """Default visualization for mse curve and mse gap curve for one test"""
    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
    
    baseline = np.array([d / (1. + np.exp(-logsnr)) for logsnr in logsnrs])
    baseline2 = t.sigmoid(logsnrs + log_eigs.view((-1, 1))).sum(axis=0).numpy()
    ax = axs[0]
    ax.plot(logsnrs, baseline, label='$N(0,1)$ MMSE')
    ax.plot(logsnrs, baseline2, lw=3, label='$N(\mu,\Sigma)$ MMSE')
    ax.plot(logsnrs, mses, label='Data MSE')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\gamma$)')
    ax.legend()
    ax.set_title('epoch=2')

    ax = axs[1]
    ax.plot(logsnrs, baseline - np.array(mses), label='MMSE Gap for $N(0,1)$')
    ax.plot(logsnrs, baseline2 - np.array(mses), label='MMSE Gap for $N(\mu,\Sigma)$')
    ax.set_ylim(-0.01, np.max(np.array(baseline) - np.array(mses)))
    ax.set_ylabel('MMSE Gap $(\epsilon)$')
    ax.set_xlabel('log SNR ($\gamma$)')
    ax.legend()

    fig.set_tight_layout(True)

    return fig

def viz_multiple(mses, log_eigs, logsnrs, d, train_loss, val_loss, nll):
    """Visualization for multiple mse curves and loss curves"""
    cmap = sns.color_palette("flare", as_cmap=True)

    length = len(mses)
    epochs = len(val_loss)
    niter = len(train_loss)

    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(6, 7))

    ax = axs[0]
    ax.plot(np.arange(1, niter+1), train_loss[:niter], '--o', label='train_loss')
    ax.plot(np.arange(0, epochs), val_loss[:epochs], '-o', label='test_loss')
    ax.plot(np.arange(0, epochs), nll[:epochs], '-o', label='test_nll')
    ax.legend()
    ax.set_ylabel('NLL (bpd)')
    ax.set_xlabel('Epochs')
    ax.set_title('DDPM')    

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
    ax.legend(fontsize = 'x-small')

def viz_change_mse(mses, log_eigs, logsnrs, d):
    """Visualization for multiple mse curves only (IDDPM and DDPM)"""
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

    # cnt = 0
    # for y, x in zip(mses[length:], logsnrs[length:]):
    #     if cnt == 0:
    #         ax.plot(x, y, label=f'before fine-tuning') 
    #     else:
    #         ax.plot(x, y, color=cmap(cnt/len(mses)), label=f'epoch{cnt}_iddpm')
    #     cnt += 1

    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\gamma$)')
    ax.set_yticks([0, d/4, d/2, 3*d/4, d], ['0', 'd/4', 'd/2', '3d/4', 'd'])
    # ax.legend(fontsize = 'x-small')
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

def viz_change_loss(train_loss, val_loss, nll):
    """Visualization for multiple loss curves only (IDDPM and DDPM)"""
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
    var = [] # add variations
    epoch = 10

    covariance = t.load('./scripts/cifar_covariance.pt')
    mu, U, log_eigs = covariance

    # load one test result 
    # x = np.load(f'/home/theo/Research_Results/debug/iid_sampler/train_bs64/ddpm_results_epoch1_bs1.npy', allow_pickle=True)
    # mses = x[0]['mses']
    # logsnrs = x[0]['logsnr']
    # print(x[0]['nll (nats)'], x[1])
    # fig = viz_one(mses, log_eigs, logsnrs, 32*32*3)
    # plt.show()

    # load ddpm mse curves and loss
    train_loss1 = np.load(f'./results/fine_tune/ddpm/train_loss_all.npy', allow_pickle=True)/32/32/3/np.log(2.0)
    train_loss.extend(train_loss1.tolist()[:epoch])
    test_loss1 = np.load(f'./results/fine_tune/ddpm/test_loss_all.npy', allow_pickle=True)/32/32/3/np.log(2.0)
    test_loss.extend(test_loss1.tolist()[:epoch+1])
    for i in range(epoch+1):
        x = np.load(f'./results/fine_tune/ddpm/results_epoch{i}.npy', allow_pickle=True)
        mses.append(x.item()['mses'])
        logsnrs.append(x.item()['logsnr'])
        nll.append(x.item()['nll (bpd)'].cpu().numpy())

    # load iddpm mse curves and loss
    train_loss2 = np.load(f'./results/fine_tune/iddpm/train_loss_all.npy', allow_pickle=True)/32/32/3/np.log(2.0)
    train_loss.extend(train_loss2.tolist()[:epoch])
    test_loss2 = np.load(f'./results/fine_tune/iddpm/test_loss_all.npy', allow_pickle=True)/32/32/3/np.log(2.0)
    test_loss.extend(test_loss2.tolist()[:epoch+1])
    for i in range(epoch+1):
        x = np.load(f'./results/fine_tune/iddpm/results_epoch{i}.npy', allow_pickle=True)
        mses.append(x.item()['mses'])
        logsnrs.append(x.item()['logsnr'])
        nll.append(x.item()['nll (bpd)'].cpu().numpy())

    print("train loss: {} \n test loss: {}".format(train_loss, test_loss))
    print('test nll:', nll)


    fig1 = viz_change_mse(mses, log_eigs, logsnrs, 32*32*3)
    fig2 = viz_change_loss(train_loss, test_loss, nll)
    # fig3 = viz_multiple(mses, log_eigs, logsnrs, 32*32*3, train_loss, test_loss, nll)
    fig1.savefig(f'./results/figs/MSE.png')
    fig2.savefig(f'./results/figs/LOSS.png')
    fig1.savefig(f'./results/figs/MSE.pdf')
    fig2.savefig(f'./results/figs/LOSS.pdf')
    # plt.show()

def process_results():
    ddpm = np.load('./results/fine_tune/ddpm/results_epoch0.npy', allow_pickle=True).item()
    iddpm = np.load('./results/fine_tune/iddpm/results_epoch0.npy', allow_pickle=True).item()
    iddpm_tune = np.load('./results/fine_tune/iddpm/results_epoch10.npy', allow_pickle=True).item()
    ddpm_tune = np.load('./results/fine_tune/ddpm/results_epoch10.npy', allow_pickle=True).item()

    # Properties of data used
    delta = 2. / 255
    d = 32 * 32 * 3
    clip = 4
    log_eigs = t.load('./scripts/cifar_covariance.pt')[2]  # Load cached spectrum for speed
    h_g = 0.5 * d * math.log(2 * math.pi * math.e) + 0.5 * log_eigs.sum().item()
    mmse_g = ddpm['mmse_g']
    logsnr = ddpm['logsnr'].numpy()
    mmse_g_1 = d / (1+np.exp(-logsnr))
    # Used to estimate good range for integration
    loc_logsnr = -log_eigs.mean().item()
    scale_logsnr = t.sqrt(1 + 3. / math.pi * log_eigs.var()).item()
    w = scale_logsnr * np.tanh(clip / 2) / (expit((logsnr - loc_logsnr)/scale_logsnr) * expit(-(logsnr - loc_logsnr)/scale_logsnr))
    w = t.tensor(w)

    def cont_bpd_from_mse(w, mses):
        nll_nats = h_g - 0.5 * (w * (mmse_g.cpu() - mses.cpu())).mean()
        return nll_nats / math.log(2) / d

    def disc_bpd_from_mse(w, mses):
        return 0.5 * (w * mses.cpu()).mean() / math.log(2) / d

    # I'm neglecting right and left tail below, but the bounds are several decimals past what we are showing.

    min_mse = reduce(t.minimum, [mmse_g, iddpm_tune['mses'], iddpm['mses'], ddpm_tune['mses'], ddpm['mses']]) 
    nll_bpd = cont_bpd_from_mse(w, min_mse)

    min_mse_discrete = reduce(t.minimum, [mmse_g, ddpm['mses'], iddpm['mses'], iddpm_tune['mses'], ddpm_tune['mses'], iddpm_tune['mses_round_xhat'], ddpm_tune['mses_round_xhat'], ddpm['mses_round_xhat'], iddpm['mses_round_xhat']]) 
    nll_bpd_discrete = disc_bpd_from_mse(w, min_mse_discrete)

    # Continuous
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 6, forward=True)
    ax.plot(logsnr, mmse_g_1, label='MMSE$_\epsilon$ for $N(0, I)$')
    ax.plot(logsnr, ddpm['mmse_g'], label='MMSE$_\epsilon$ for $N(\\mu, \\Sigma)$')
    ax.plot(logsnr, ddpm['mses'], label='DDPM')
    ax.plot(logsnr, iddpm['mses'], label='IDDPM')
    ax.plot(logsnr, ddpm_tune['mses'], label='DDPM-tuned')
    ax.plot(iddpm_tune['logsnr'], iddpm_tune['mses'], label='IDDPM-tuned')
    ax.set_xlabel('$\\alpha$ (log SNR)')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon(z_\\alpha, \\alpha))^2]$')
    ax.fill_between(logsnr, min_mse, ddpm['mmse_g'], alpha=0.1)
    ax.set_yticks([0, d/4, d/2, 3*d/4, d], ['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.legend(loc='lower right')
    
    fig.set_tight_layout(True)
    fig.savefig('./results/figs/cont_density.pdf')
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)

    # Discrete
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 6, forward=True)
    ax.plot(logsnr, ddpm['mses'], label='DDPM')
    ax.plot(logsnr, iddpm['mses'], label='IDDPM')
    ax.plot(logsnr, ddpm_tune['mses'], label='DDPM-tuned')
    ax.plot(iddpm_tune['logsnr'], iddpm_tune['mses'], label='IDDPM-tuned')
    ax.plot(logsnr, ddpm['mses_round_xhat'], label='round(DDPM)')
    ax.plot(logsnr, iddpm['mses_round_xhat'], label='round(IDDPM)')
    ax.plot(logsnr, ddpm_tune['mses_round_xhat'], label='round(DDPM-tuned)')
    ax.plot(iddpm_tune['logsnr'], iddpm_tune['mses_round_xhat'], label='round(IDDPM-tuned)')
    ax.set_xlabel('$\\alpha$ (log SNR)')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon(z_\\alpha, \\alpha))^2]$')
    ax.fill_between(logsnr, min_mse_discrete, alpha=0.1)
    ax.set_yticks([0, d/4, d/2, 3*d/4, d], ['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.legend(loc='upper left')
    
    fig.set_tight_layout(True)
    fig.savefig('./results/figs/disc_density.pdf')

    # Output table numbers
    print('DDPM &  &  & {:.2f} \\\\'.format(ddpm['nll (bpd)']))
    print('IDDPM &  &  & {:.2f} \\\\'.format(iddpm['nll (bpd)']))
    print('\\midrule')
    print('DDPM(tune)&  &  & {:.2f} \\\\'.format(ddpm_tune['nll (bpd)']))
    print('IDDPM(tune) &  &  & {:.2f} \\\\'.format(iddpm_tune['nll (bpd)']))
    print('\\midrule')
    print('Ensemble &  &  & {:.2f} \\\\'.format(nll_bpd))

    print('\n\n\nDiscrete')
    min_deq = reduce(t.minimum, [mmse_g, ddpm['mses_dequantize'], iddpm['mses_dequantize'], ddpm_tune['mses_dequantize'], iddpm_tune['mses_dequantize']])
    ens_deq = cont_bpd_from_mse(w, min_deq) + np.log(127.5)/np.log(2)
    print('DDPM &  & {:.2f} & {:.2f} \\\\'.format(ddpm['nll-discrete (bpd)'], ddpm['nll-discrete-limit (bpd) - dequantize']))
    print('IDDPM &  & {:.2f} & {:.2f} \\\\'.format(iddpm['nll-discrete (bpd)'], iddpm['nll-discrete-limit (bpd) - dequantize']))
    print('\\midrule')
    print('DDPM(tune)&  & {:.2f} & {:.2f} \\\\'.format(ddpm_tune['nll-discrete (bpd)'],ddpm_tune['nll-discrete-limit (bpd) - dequantize']))
    print('IDDPM(tune) &  & {:.2f} & {:.2f} \\\\'.format(iddpm_tune['nll-discrete (bpd)'], iddpm_tune['nll-discrete-limit (bpd) - dequantize']))
    print('\\midrule')
    print('Ensemble &  &  {:.2f} & {:.2f} \\\\'.format(nll_bpd_discrete, ens_deq))

    for wbar, m in [(w, ddpm), (w, iddpm), (w, ddpm_tune), (w, iddpm_tune)]:
        print('Get BPDs for rounded solutions alone')
        print(disc_bpd_from_mse(wbar, m['mses_round_xhat']))

    # calculate variants
    print('\n\n\n standard deviation')
    def calc_std(var, n, d):
        return math.sqrt(var / n) / d / math.log(2.0)
    ddpm_nll_std = calc_std(ddpm['nll (nats) var'], 100, d)
    ddpm_nll_dequantize_std = calc_std(ddpm['nll (nats) - dequantize var'], 100, d)
    ddpm_nll_discrete_std = calc_std(ddpm['nll-discrete var'], 100, d)

    ddpm_tune_nll_std = calc_std(ddpm_tune['nll (nats) var'], 100, d)
    ddpm_tune_nll_dequantize_std = calc_std(ddpm_tune['nll (nats) - dequantize var'], 100, d)
    ddpm_tune_nll_discrete_std = calc_std(ddpm_tune['nll-discrete var'], 100, d)

    iddpm_nll_std = calc_std(iddpm['nll (nats) var'], 100, d)
    iddpm_nll_dequantize_std = calc_std(iddpm['nll (nats) - dequantize var'], 100, d)
    iddpm_nll_discrete_std = calc_std(iddpm['nll-discrete var'], 100, d)

    iddpm_tune_nll_std = calc_std(iddpm_tune['nll (nats) var'], 100, d)
    iddpm_tune_nll_dequantize_std = calc_std(iddpm_tune['nll (nats) - dequantize var'], 100, d)
    iddpm_tune_nll_discrete_std = calc_std(iddpm_tune['nll-discrete var'], 100, d)

    print('DDPM - nll (bpd) std: {:.2f}, nll-discrete (bpd) std: {:.2f}, nll (bpd) - dequantize std: {:.2f}'.format(ddpm_nll_std, ddpm_nll_discrete_std, ddpm_nll_dequantize_std))
    print('IDDPM - nll (bpd) std: {:.2f}, nll-discrete (bpd) std: {:.2f}, nll (bpd) - dequantize std: {:.2f}'.format(iddpm_nll_std, iddpm_nll_discrete_std, iddpm_nll_dequantize_std))
    print('DDPM-tune - nll (bpd) std: {:.2f}, nll-discrete (bpd) std: {:.2f}, nll (bpd) - dequantize std: {:.2f}'.format(ddpm_tune_nll_std, ddpm_tune_nll_discrete_std, ddpm_tune_nll_dequantize_std))
    print('IDDPM-tune - nll (bpd) std: {:.2f}, nll-discrete (bpd) std: {:.2f}, nll (bpd) - dequantize std: {:.2f}'.format(iddpm_tune_nll_std, iddpm_tune_nll_discrete_std, iddpm_tune_nll_dequantize_std))

def main():
    # plot_mse_loss()
    process_results()


if __name__ == "__main__":
    main()