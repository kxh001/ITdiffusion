import torch as t
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from tqdm import tqdm
import math
plt.style.use('seaborn-paper')

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

    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\gamma$)')
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
    ax.plot(np.arange(1, niter+1), train_loss[:niter], '--o', label='train_loss')
    ax.plot(np.arange(0, epochs), val_loss[:epochs], '-o', label='test_loss')
    ax.plot(np.arange(0, epochs), nll[:epochs], '-o', label='test_nll')
    ax.legend()
    ax.set_ylabel('NLL (bpd)')
    ax.set_xlabel('Epochs')
    ax.set_title('DDPM')

    ax = axs[0]
    ax.plot(np.arange(1, niter+1), train_loss[niter:], '--o', label='train_loss')
    ax.plot(np.arange(0, epochs), val_loss[epochs:], '-o', label='test_loss')
    ax.plot(np.arange(0, epochs), nll[epochs:], '-o', label='test_nll')
    ax.legend()
    ax.set_ylabel('NLL (bpd)')
    ax.set_xlabel('Epochs')
    ax.set_title('IDDPM')
    fig.set_tight_layout(True)

    return fig



def main():
    mses = []
    logsnrs = []
    train_loss = []
    test_loss = []
    nll = []
    epoch = 5

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
    train_loss1 = np.load('./results/debug/iid_sampler/train_bs64/ddpm_train_loss_all.npy', allow_pickle=True)/32/32/3/np.log(2.0)
    train_loss.extend(train_loss1.tolist()[:epoch])
    for i in range(epoch+1):
        # x = np.load(f'./results/fine_tune/ddpm/results_epoch{i}.npy', allow_pickle=True)
        x = np.load(f'./results/debug/iid_sampler/train_bs64/ddpm_results_epoch{i}_base.npy', allow_pickle=True)
        if i == 0:
            mses.append(x[0]['mses'])
            logsnrs.append(x[0]['logsnr'])
            test_loss.append(x[1]/32/32/3/np.log(2.0))
            nll.append(x[0]['nll (bpd)'].cpu().numpy())
        else:
            mses.append(x[0]['mses'])
            logsnrs.append(x[0]['logsnr'])
            # y = np.load(f'./results/fine_tune/ddpm/train_loss_epoch{i}.npy', allow_pickle=True)
            # train_loss.append(y/32/32/3/np.log(2.0))
            test_loss.append(x[1]/32/32/3/np.log(2.0))
            nll.append(x[0]['nll (bpd)'].cpu().numpy())

    # load iddpm mse curves and loss
    train_loss2 = np.load('./results/debug/iid_sampler/train_bs32/iddpm_train_loss_all.npy', allow_pickle=True)/32/32/3/np.log(2.0)
    train_loss.extend(train_loss2.tolist()[:epoch])
    for i in range(epoch+1):
        # x = np.load(f'./results/fine_tune/iddpm/results_epoch{i}.npy', allow_pickle=True)
        x = np.load(f'./results/debug/iid_sampler/train_bs32/iddpm_results_epoch{i}_base.npy', allow_pickle=True)
        if i == 0:
            mses.append(x[0]['mses'])
            logsnrs.append(x[0]['logsnr'])
            test_loss.append(x[1]/32/32/3/np.log(2.0))
            nll.append(x[0]['nll (bpd)'].cpu().numpy())
        else:
            mses.append(x[0]['mses'])
            logsnrs.append(x[0]['logsnr'])
            # y = np.load(f'./results/fine_tune/iddpm/train_loss_epoch{i}.npy', allow_pickle=True)
            # train_loss.append(y/32/32/3/np.log(2.0))
            test_loss.append(x[1]/32/32/3/np.log(2.0))
            nll.append(x[0]['nll (bpd)'].cpu().numpy())
    print("train loss: {} \n test loss: {}".format(train_loss, test_loss))


    fig1 = viz_change_mse(mses, log_eigs, logsnrs, 32*32*3)
    fig2 = viz_change_loss(train_loss, test_loss, nll)
    # fig3 = viz_multiple(mses, log_eigs, logsnrs, 32*32*3, train_loss, test_loss, nll)
    fig1.savefig(f'./results/figs/MSE.pdf')
    fig2.savefig(f'./results/figs/LOSS.pdf')
    plt.show()


if __name__ == "__main__":
    main()