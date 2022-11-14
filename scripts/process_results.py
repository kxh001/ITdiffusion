import numpy as np
import torch as t
import sys

sys.path.append('..')
from utilsiddpm.utils import logistic_integrate
from functools import reduce
import matplotlib.pyplot as plt
import math
plt.style.use('seaborn-paper')
import seaborn as sns
# sns.set(font_scale=2)
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})


ddpm = np.load('paper_results/results_ddpm_high.npy', allow_pickle=True).item()
iddpm = np.load('paper_results/results_iddpm.npy', allow_pickle=True).item()
iddpm_tune = np.load('paper_results/results_iddpm_fine-tune.npy', allow_pickle=True)[0]
ddpm_tune = np.load('paper_results/results_ddpm_low_fine-tune.npy', allow_pickle=True)[0]

# Properties of data used
delta = 2. / 255
d = 32 * 32 * 3
log_eigs = t.load('cifar_covariance.pt')[2]  # Load cached spectrum for speed
h_g = 0.5 * d * math.log(2 * math.pi * math.e) + 0.5 * log_eigs.sum().item()
# Used to estimate good range for integration
loc_logsnr = -log_eigs.mean().item()
scale_logsnr = t.sqrt(1 + 3. / math.pi * log_eigs.var()).item()
logsnr2, w = logistic_integrate(100, loc=loc_logsnr, scale=scale_logsnr, clip=4, deterministic=True)

logsnr_t, w_t = logistic_integrate(npoints=100, loc=9., scale=4., clip=4., device='cpu', deterministic=True)

fig, ax = plt.subplots(1)
# ax.set_aspect('equal')
fig.set_size_inches(10, 6, forward=True)


mmse_g = ddpm['mmse_g']
def cont_bpd_from_mse(w, mses):
    nll_nats = h_g - 0.5 * (w * (mmse_g.cpu() - mses.cpu())).mean()
    return nll_nats / math.log(2) / d

def disc_bpd_from_mse(w, mses):
    return 0.5 * (w * mses.cpu()).mean() / math.log(2) / d

# I'm neglecting right and left tail below, but the bounds are several decimals past what we are showing.

# min_mse = reduce(t.minimum, [ddpm['mses'], iddpm['mses'], iddpm_tune['mses'], ddpm_tune['mses'], mmse_g])  # add ddpm_tune
min_mse = reduce(t.minimum, [iddpm_tune['mmse_g'], iddpm_tune['mses']])  # add ddpm_tune
nll_bpd = cont_bpd_from_mse(w_t, min_mse)

#min_mse_discrete = reduce(t.minimum, [ddpm['mses'], iddpm['mses'], iddpm_tune['mses'], ddpm_tune['mses'], ddpm_tune['mses_round_xhat'], ddpm['mses_round_xhat'], iddpm['mses_round_xhat'], iddpm_tune['mses_round_xhat'], mmse_g])  # add ddpm_tune
min_mse_discrete = reduce(t.minimum, [iddpm_tune['mmse_g'], iddpm_tune['mses'], iddpm_tune['mses_round_xhat']])  # add ddpm_tune

nll_bpd_discrete = disc_bpd_from_mse(w_t, min_mse_discrete)



logsnr = ddpm['logsnr'].numpy()
mmse_g_1 = d / (1+np.exp(-logsnr))
ax.plot(logsnr, mmse_g_1, label='MMSE$_\epsilon$ for $N(0, I)$')
ax.plot(logsnr, ddpm['mmse_g'], label='MMSE$_\epsilon$ for $N(\\mu, \\Sigma)$')
ax.plot(logsnr, ddpm['mses'], label='DDPM')
ax.plot(logsnr, iddpm['mses'], label='IDDPM')
# ax.plot(logsnr, ddpm_tune['mses'], label='DDPM-tuned')
ax.plot(iddpm_tune['logsnr'], iddpm_tune['mses'], label='IDDPM-tuned')
ax.set_xlabel('$\\alpha$ (log SNR)')
ax.set_ylabel('$E[(\epsilon - \hat \epsilon(z_\\alpha, \\alpha))^2]$')
ax.fill_between(logsnr, min_mse, ddpm['mmse_g'], alpha=0.1)
ax.set_yticks([0, d/4, d/2, 3*d/4, d], ['0', 'd/4', 'd/2', '3d/4', 'd'])


ax.legend(loc='lower right')
fig.set_tight_layout(True)
fig.savefig('../figures/cont_density.pdf')
# fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)

# Discrete
fig, ax = plt.subplots(1)
# ax.set_aspect('equal')
fig.set_size_inches(10, 6, forward=True)

ax.plot(logsnr, ddpm['mses'], label='DDPM')
ax.plot(logsnr, iddpm['mses'], label='IDDPM')
# ax.plot(logsnr, ddpm_tune['mses'], label='DDPM-tuned')
ax.plot(iddpm_tune['logsnr'], iddpm_tune['mses'], label='IDDPM-tuned')

ax.plot(logsnr, ddpm['mses_round_xhat'], label='round(DDPM)')
ax.plot(logsnr, iddpm['mses_round_xhat'], label='round(IDDPM)')
# ax.plot(ddpm['logsnr'], ddpm_tune['mses_round_xhat'], label='round(DDPM-tuned)')
# ax.plot(logsnr, ddpm_tune['mses_round_xhat'], label='round(DDPM-tuned)')
ax.plot(iddpm_tune['logsnr'], iddpm_tune['mses_round_xhat'], label='round(IDDPM-tuned)')
ax.set_xlabel('$\\alpha$ (log SNR)')
ax.set_ylabel('$E[(\epsilon - \hat \epsilon(z_\\alpha, \\alpha))^2]$')
ax.fill_between(logsnr, min_mse_discrete, alpha=0.1)
ax.set_yticks([0, d/4, d/2, 3*d/4, d], ['0', 'd/4', 'd/2', '3d/4', 'd'])

ax.legend(loc='upper left')
fig.set_tight_layout(True)
fig.savefig('../figures/disc_density.pdf')

# Output table numbers


print('DDPM &  &  & {:.2f} \\\\'.format(ddpm['nll (bpd)']))
print('IDDPM &  &  & {:.2f} \\\\'.format(iddpm['nll (bpd)']))
print('\\midrule')
print('DDPM(tune)&  &  & {:.2f} \\\\'.format(ddpm_tune['nll (bpd)']))
print('IDDPM(tune) &  &  & {:.2f} \\\\'.format(iddpm_tune['nll (bpd)']))
print('\\midrule')

print('Ensemble &  &  & {:.2f} \\\\'.format(nll_bpd))

print('\n\n\nDiscrete')
# min_deq = reduce(t.minimum, [mmse_g, ddpm['mses_dequantize'], iddpm['mses_dequantize'], ddpm_tune['mses_dequantize'], iddpm_tune['mses_dequantize']])
min_deq = reduce(t.minimum, [iddpm_tune['mmse_g'], iddpm_tune['mses_dequantize']])
ens_deq = cont_bpd_from_mse(w_t, min_deq) + np.log(127.5)/np.log(2)
print('DDPM &  & {:.2f} & {:.2f} \\\\'.format(ddpm['nll-discrete (bpd)'], ddpm['nll-discrete-limit (bpd) - dequantize']))
print('IDDPM &  & {:.2f} & {:.2f} \\\\'.format(iddpm['nll-discrete (bpd)'], iddpm['nll-discrete-limit (bpd) - dequantize']))
print('\\midrule')
print('DDPM(tune)&  & {:.2f} & {:.2f} \\\\'.format(ddpm_tune['nll-discrete (bpd)'],ddpm_tune['nll-discrete-limit (bpd) - dequantize']))
print('IDDPM(tune) &  & {:.2f} & {:.2f} \\\\'.format(iddpm_tune['nll-discrete (bpd)'], iddpm_tune['nll-discrete-limit (bpd) - dequantize']))
print('\\midrule')

print('Ensemble &  &  {:.2f} & {:.2f} \\\\'.format(nll_bpd_discrete, ens_deq))

for wbar, m in [(w, ddpm), (w, iddpm), (w, ddpm_tune), (w_t, iddpm_tune)]:
    print('Get BPDs for rounded solutions alone')
    print(disc_bpd_from_mse(wbar, m['mses_round_xhat']))


import IPython; IPython.embed()