import torch as t
import matplotlib.pyplot as plt
from scipy.special import expit
from matplotlib import cm
import seaborn as sns
import numpy as np
import sys
import math
from utilsiddpm.utils import logistic_integrate
from functools import reduce
plt.style.use('seaborn-paper')
sys.path.append('..')
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})

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
    ddpm = np.load('./results/fine_tune/ddpm_soft/results_epoch0.npy', allow_pickle=True)[0]
    iddpm = np.load('./results/fine_tune/iddpm_soft/results_epoch0.npy', allow_pickle=True)[0]
    ddpm_tune = np.load('./results/fine_tune/ddpm_soft/results_epoch10.npy', allow_pickle=True)[0]
    iddpm_tune = np.load('./results/fine_tune/iddpm_soft/results_epoch10.npy', allow_pickle=True)[0]
    # iddpm_tune_soft = np.load('./results/variance/iddpm-softUNet/results_epoch10.npy', allow_pickle=True)[0]
    # ddpm_tune_soft = np.load('./results/variance/ddpm-softUNet/results_epoch10.npy', allow_pickle=True)[0]

    import IPython; IPython.embed()

    # Properties of data used
    delta = 2. / 255
    d = 32 * 32 * 3
    clip = 4
    log_eigs = t.load('./scripts/cifar_covariance.pt')[2]  # Load cached spectrum for speed
    h_g = 0.5 * d * math.log(2 * math.pi * math.e) + 0.5 * log_eigs.sum().item()
    mmse_g = ddpm['mmse_g']
    logsnr = ddpm['logsnr'] # With the same random seed on the same device, logsnrs are the same
    mmse_g_1 = d / (1+np.exp(-logsnr))
    # Used to estimate good range for integration
    loc_logsnr = -log_eigs.mean().item()
    scale_logsnr = t.sqrt(1 + 3. / math.pi * log_eigs.var()).item()
    w = scale_logsnr * np.tanh(clip / 2) / (expit((logsnr - loc_logsnr)/scale_logsnr) * expit(-(logsnr - loc_logsnr)/scale_logsnr))
    w = t.tensor(w)

    def cont_bpd_from_mse(w, mses):
        nll_nats = h_g - 0.5 * (w * (mmse_g.cpu() - mses.cpu())).mean()
        return nll_nats / math.log(2) / d

    # def disc_bpd_from_mse(logsnr, mses):
    #     return 0.5 * t.trapz(mses, logsnr) / math.log(2) / d

    def disc_bpd_from_mse(w, mses):
        return 0.5 * (w * mses.cpu()).mean() / math.log(2) / d

    # I'm neglecting right and left tail below, but the bounds are several decimals past what we are showing.
    
    min_mse = reduce(t.minimum, [mmse_g, iddpm_tune['mses'], iddpm['mses'], ddpm_tune['mses'], ddpm['mses']]) 
    nll_bpd = cont_bpd_from_mse(w, min_mse)

    min_mse_discrete = reduce(t.minimum, [mmse_g, ddpm['mses'], iddpm['mses'], iddpm_tune['mses'], ddpm_tune['mses'], ddpm['mses_round_xhat'], iddpm['mses_round_xhat']]) #  iddpm_tune_soft['mses_round_xhat'], ddpm_tune_soft['mses_round_xhat']
    nll_bpd_discrete = disc_bpd_from_mse(w, min_mse_discrete)

    min_deq = reduce(t.minimum, [mmse_g, ddpm['mses_dequantize'], iddpm['mses_dequantize'], ddpm_tune['mses_dequantize'], iddpm_tune['mses_dequantize']])
    ens_deq = cont_bpd_from_mse(w, min_deq) + np.log(127.5)/np.log(2)

    cmap2 = sns.color_palette("Paired")
    cmap = sns.color_palette()
    # Continuous
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
    ax.set_yticks([0, d/4, d/2, 3*d/4, d])
    ax.set_yticklabels(['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.legend(loc='lower right')
    
    fig.set_tight_layout(True)
    # fig.savefig('./results/figs/cont_density.pdf')
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)

    # Discrete
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 6, forward=True)
    ax.plot(logsnr, ddpm['mses'], label='DDPM', color=cmap2[0])
    ax.plot(logsnr, iddpm['mses'], label='IDDPM', color=cmap2[2])
    ax.plot(logsnr, ddpm_tune['mses'], label='DDPM-tuned', color=cmap2[1])
    ax.plot(logsnr, iddpm_tune['mses'], label='IDDPM-tuned', color=cmap2[3])
    ax.plot(logsnr, ddpm['mses_round_xhat'],label='round(DDPM)', color=cmap2[5])
    ax.plot(logsnr, iddpm['mses_round_xhat'],label='round(IDDPM)', color=cmap2[9])
    ax.plot(logsnr, ddpm_tune['mses_round_xhat'], '--',label='round(DDPM-tuned)', color=cmap2[4])
    ax.plot(logsnr, iddpm_tune['mses_round_xhat'], '--', label='round(IDDPM-tuned)', color=cmap2[8])

    # ax.plot(ddpm_tune_soft['logsnr'], ddpm_tune_soft['mses_round_xhat'], label='round(DDPM-tuned)')
    # ax.plot(iddpm_tune_soft['logsnr'], iddpm_tune_soft['mses_round_xhat'], label='round(IDDPM-tuned)')

    ax.set_xlabel('$\\alpha$ (log SNR)')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon(z_\\alpha, \\alpha))^2]$')
    ax.fill_between(logsnr, min_mse_discrete, alpha=0.1)
    ax.set_yticks([0, d/4, d/2, 3*d/4, d])
    ax.set_yticklabels(['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.legend(loc='upper left')

    fig.set_tight_layout(True)
    # fig.savefig('./results/figs/disc_density.pdf')
    plt.show()


    # Output table numbers
    print('Continuous')
    print('DDPM &  &  & {:.2f} \\\\'.format(ddpm['nll (bpd)']))
    print('IDDPM &  &  & {:.2f} \\\\'.format(iddpm['nll (bpd)']))
    print('\\midrule')
    print('DDPM(tune)&  &  & {:.2f} \\\\'.format(ddpm_tune['nll (bpd)']))
    print('IDDPM(tune) &  &  & {:.2f} \\\\'.format(iddpm_tune['nll (bpd)']))
    print('\\midrule')
    print('Ensemble &  &  & {:.2f} \\\\'.format(nll_bpd))

    print('\n\n\nDiscrete')
    print('DDPM &  & {:.2f} & {:.2f} \\\\'.format(disc_bpd_from_mse(w, ddpm['mses_round_xhat']), ddpm['nll-discrete-limit (bpd) - dequantize']))
    print('IDDPM &  & {:.2f} & {:.2f} \\\\'.format(disc_bpd_from_mse(w, iddpm['mses_round_xhat']), iddpm['nll-discrete-limit (bpd) - dequantize']))
    print('\\midrule')
    print('DDPM(tune)&  & {:.2f} & {:.2f} \\\\'.format(disc_bpd_from_mse(w, ddpm_tune['mses_round_xhat']),ddpm_tune['nll-discrete-limit (bpd) - dequantize']))
    print('IDDPM(tune) &  & {:.2f} & {:.2f} \\\\'.format(disc_bpd_from_mse(w, iddpm_tune['mses_round_xhat']), iddpm_tune['nll-discrete-limit (bpd) - dequantize']))
    print('\\midrule')
    print('Ensemble &  &  {:.2f} & {:.2f} \\\\'.format(nll_bpd_discrete, ens_deq))

    # variants
    print('DDPM - nll (bpd) std: {:.5f}, nll-discrete (bpd) std: {:.5f}, nll (bpd) - dequantize std: {:.5f}'.format(ddpm['nll (bpd) - std'], ddpm['nll-discrete (bpd) - std'], ddpm['nll (bpd) - dequantize - std']))
    print('IDDPM - nll (bpd) std: {:.5f}, nll-discrete (bpd) std: {:.5f}, nll (bpd) - dequantize std: {:.5f}'.format(iddpm['nll (bpd) - std'], iddpm['nll-discrete (bpd) - std'], iddpm['nll (bpd) - dequantize - std']))
    print('DDPM-tune - nll (bpd) std: {:.5f}, nll-discrete (bpd) std: {:.5f}, nll (bpd) - dequantize std: {:.5f}'.format(ddpm_tune['nll (bpd) - std'], ddpm_tune['nll-discrete (bpd) - std'], ddpm_tune['nll (bpd) - dequantize - std']))
    print('IDDPM-tune - nll (bpd) std: {:.5f}, nll-discrete (bpd) std: {:.5f}, nll (bpd) - dequantize std: {:.5f}'.format(iddpm_tune['nll (bpd) - std'], iddpm_tune['nll-discrete (bpd) - std'], iddpm_tune['nll (bpd) - dequantize - std']))

def process_results_disc():
    ddpm = np.load('./results/npoints/ddpm/results_epoch0.npy', allow_pickle=True)[0]
    iddpm = np.load('./results/npoints/iddpm/results_epoch0.npy', allow_pickle=True)[0]
    ddpm_tune = np.load('./results/npoints/ddpm/results_epoch10.npy', allow_pickle=True)[0]
    iddpm_tune = np.load('./results/npoints/iddpm/results_epoch10.npy', allow_pickle=True)[0]

    # import IPython; IPython.embed()

    # Properties of data used
    delta = 2. / 255
    d = 32 * 32 * 3
    clip = 4
    log_eigs = t.load('./scripts/cifar_covariance.pt')[2]  # Load cached spectrum for speed
    logsnr = iddpm['logsnr'] # With the same random seed on the same device, logsnrs are the same
    w = iddpm['w']

    # Used to estimate good range for integration
    # loc_logsnr = -log_eigs.mean().item()
    # scale_logsnr = t.sqrt(1 + 3. / math.pi * log_eigs.var()).item()
    # w = scale_logsnr * np.tanh(clip / 2) / (expit((logsnr - loc_logsnr)/scale_logsnr) * expit(-(logsnr - loc_logsnr)/scale_logsnr))
    # w = t.tensor(w)


    def disc_bpd_from_mse(w, mses):
        return 0.5 * (w * mses.cpu()).mean() / math.log(2) / d

    # I'm neglecting right and left tail below, but the bounds are several decimals past what we are showing.

    min_mse_discrete = reduce(t.minimum, [ddpm['mses'], iddpm['mses'], iddpm_tune['mses'], ddpm_tune['mses'], ddpm['mses_round_xhat'], iddpm['mses_round_xhat']]) #  iddpm_tune_soft['mses_round_xhat'], ddpm_tune_soft['mses_round_xhat']
    nll_bpd_discrete = disc_bpd_from_mse(w, min_mse_discrete)

    cmap2 = sns.color_palette("Paired")
    cmap = sns.color_palette()

    # Discrete
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 6, forward=True)
    ax.plot(logsnr, ddpm['mses'], label='DDPM', color=cmap2[0])
    ax.plot(logsnr, iddpm['mses'], label='IDDPM', color=cmap2[2])
    ax.plot(logsnr, ddpm_tune['mses'], label='DDPM-tuned', color=cmap2[1])
    ax.plot(logsnr, iddpm_tune['mses'], label='IDDPM-tuned', color=cmap2[3])
    ax.plot(logsnr, ddpm['mses_round_xhat'],label='round(DDPM)', color=cmap2[5])
    ax.plot(logsnr, iddpm['mses_round_xhat'],label='round(IDDPM)', color=cmap2[9])
    ax.plot(logsnr, ddpm_tune['mses_round_xhat'], '--',label='round(DDPM-tuned)', color=cmap2[4])
    ax.plot(logsnr, iddpm_tune['mses_round_xhat'], '--', label='round(IDDPM-tuned)', color=cmap2[8])

    # ax.plot(ddpm_tune_soft['logsnr'], ddpm_tune_soft['mses_round_xhat'], label='round(DDPM-tuned)')
    # ax.plot(iddpm_tune_soft['logsnr'], iddpm_tune_soft['mses_round_xhat'], label='round(IDDPM-tuned)')

    ax.set_xlabel('$\\alpha$ (log SNR)')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon(z_\\alpha, \\alpha))^2]$')
    ax.fill_between(logsnr, min_mse_discrete, alpha=0.1)
    ax.set_yticks([0, d/4, d/2, 3*d/4, d])
    ax.set_yticklabels(['0', 'd/4', 'd/2', '3d/4', 'd'])
    ax.legend(loc='upper left')

    fig.set_tight_layout(True)
    # fig.savefig('./results/figs/disc_density.pdf')
    plt.show()


    # Output table numbers
    print('Discrete')
    print('DDPM &  & {:.2f} &  \\\\'.format(disc_bpd_from_mse(w, ddpm['mses_round_xhat'])))
    print('IDDPM &  & {:.2f} &  \\\\'.format(disc_bpd_from_mse(w, iddpm['mses_round_xhat'])))
    print('\\midrule')
    print('DDPM(tune)&  & {:.2f} &  \\\\'.format(disc_bpd_from_mse(w, ddpm_tune['mses_round_xhat'])))
    print('IDDPM(tune) &  & {:.2f} &  \\\\'.format(disc_bpd_from_mse(w, iddpm_tune['mses_round_xhat'])))
    print('\\midrule')
    print('Ensemble &  &  {:.2f} &  \\\\'.format(nll_bpd_discrete))

    # variants
    print('DDPM - nll-discrete (bpd) std: {:.5f}'.format(ddpm['nll-discrete (bpd) - std']))
    print('IDDPM - nll-discrete (bpd) std: {:.5f}'.format(iddpm['nll-discrete (bpd) - std']))
    print('DDPM-tune - nll-discrete (bpd) std: {:.5f}'.format(ddpm_tune['nll-discrete (bpd) - std']))
    print('IDDPM-tune - nll-discrete (bpd) std: {:.5f}'.format(iddpm_tune['nll-discrete (bpd) - std']))

def main():
    # plot_mse_loss()
    # process_results()
    process_results_disc()


if __name__ == "__main__":
    main()