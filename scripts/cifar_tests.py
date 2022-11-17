"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os
import numpy as np
import torch.distributed as dist
import torch as t
import torchvision as tv
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.animation as animation
from matplotlib import collections as mc
from matplotlib.collections import LineCollection
from tqdm import tqdm

import sys
sys.path.append('..')  # Lame, must be a better way to do relative import
import diffusionmodel as dm
# from utilsiddpm.utils import viz, plot_image
from utilsiddpm import dist_util, logger
from utilsiddpm.image_datasets import load_data, load_dataloader

from utilsiddpm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # logger.log("creating model and diffusion...")
    # model = create_model(args.image_size,
    #     args.num_channels,
    #     args.num_res_blocks,
    #     learn_sigma=args.learn_sigma,
    #     class_cond=args.class_cond,
    #     use_checkpoint=args.use_checkpoint,
    #     attention_resolutions=args.attention_resolutions,
    #     num_heads=args.num_heads,
    #     num_heads_upsample=args.num_heads_upsample,
    #     use_scale_shift_norm=args.use_scale_shift_norm,
    #     dropout=args.dropout,
    #     wrapped=True)
    #
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    # model.to(dist_util.dev())
    # model.eval()

    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,  # Generating MMSE curves with subset, better to use random subset
        # subset=100
    )

    # # I changed the hard coded train location for my system, should come up with better solution
    train_loc = '../data/cifar_train'
    data_train = load_data(
        data_dir = train_loc,
        batch_size = 50000,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("calculate integral bound...")
    diffusion = dm.DiffusionModel(model)
    covariance = t.load('cifar_covariance.pt')  # Load cached spectrum for speed
    covariance = [q.to(dist_util.dev()) for q in covariance]
    log_eigs = covariance[2]
    diffusion.dataset_info(data_train, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")

    logger.log("Testing test_nll code")
    # results = diffusion.test_nll(data, npoints=100, delta=1./127.5, xinterval=(-1, 1))
    # print(results)
    # with t.no_grad():
    #     for batch in tqdm(data):
    #         result_nll = diffusion.nll_disc([batch[0].to('cuda')], logsnr_samples_per_x=100, xinterval=(-1,1), delta=1./127.5, soft=True)
    # import IPython; IPython.embed()
    results, val_loss = diffusion.test_nll_disc(data, npoints=1000, delta=1./127.5, xinterval=(-1, 1), soft=True, max_x_samples=1000)
    print(results)
#    results2, val_loss = diffusion.test_nll(data, npoints=100, delta=1./127.5, xinterval=(-1, 1), soft=True)
#    print(results2)
    import IPython; IPython.embed()

    logger.log("Generating samples")
    # TODO: Add a sampling test

    # schedule = diffusion.generate_schedule(1000, plot_grid=True)
    schedule = t.load('schedule_cifar_500.pt')
    # schedule = diffusion.generate_schedule(100, data, plot_grid=True)  # Pretty slow! Maybe an hour or two - subset of data is sufficient, and we can store it

    z = diffusion.sample(schedule, n_samples=9, store_t=5)
    image_movie(z, '../figures/sample_test.mp4')

    import IPython;
    IPython.embed()

    # Reload full test dataset for NLL test.
    data = load_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False
    )

    # Look at MSE curves for images and non-images
    batch = next(data)[0].to(dist_util.dev())
    fig = mmse_x_curve(diffusion, batch[0])
    fig.savefig('../figures/mse_real_image.png')

    logsnrs = t.linspace(-5, 15, 20, device=batch.device)
    zs = diffusion.noisy_channel(batch[0], logsnrs)  # noisy version of real image
    fig = mmse_x_curve(diffusion, zs[0][0])
    fig.savefig('../figures/mse_lowsnr_real_image.png')
    fig = mmse_x_curve(diffusion, zs[0][-1])
    fig.savefig('../figures/mse_highsnr_real_image.png')

    # MSE for an image drawn from matching Gaussian
    zn = diffusion.sample_g()
    fig = mmse_x_curve(diffusion, zn)
    fig.savefig('../figures/mse_gaussian_image.png')

    # Look at how using diffusion to update state changes MSE curves
    snr = t.tensor([1.], device=zn.device)
    zn_eps = diffusion([zn.unsqueeze(0)], snr)
    zn2 = (t.sqrt(1 + snr) * zn - zn_eps[0]) / t.sqrt(snr)
    fig = mmse_x_curve(diffusion, zn2)
    fig.savefig('../figures/mse_gaussian_image_with_one_step_snr1.png')

    snr = t.tensor([5.], device=zn.device)
    zn_eps = diffusion([zn.unsqueeze(0)], snr)
    zn2 = (t.sqrt(1 + snr) * zn - zn_eps[0]) / t.sqrt(snr)
    fig = mmse_x_curve(diffusion, zn2)
    fig.savefig('../figures/mse_gaussian_image_with_one_step_snr5.png')

    snr = t.tensor([10.], device=zn.device)
    zn_eps = diffusion([zn.unsqueeze(0)], snr)
    zn2 = (t.sqrt(1 + snr) * zn - zn_eps[0]) / t.sqrt(snr)
    fig = mmse_x_curve(diffusion, zn2)
    fig.savefig('../figures/mse_gaussian_image_with_one_step_snr10.png')

    snr = t.tensor([1000.], device=zn.device)
    zn_eps = diffusion([zn.unsqueeze(0)], snr)
    zn2 = (t.sqrt(1 + snr) * zn - zn_eps[0]) / t.sqrt(snr)
    fig = mmse_x_curve(diffusion, zn2)
    fig.savefig('../figures/mse_gaussian_image_with_one_step_snr1000.png')




    logger.log("Generating MSE curve")
    # high quality MSE on snr grid
    logsnr_grid = t.linspace(-6, 16, 60)
    mse_grid = t.zeros(len(logsnr_grid))
    num_batches = 20   # CIFAR test set is 10000
    logger.log(f"Batch size: {args.batch_size}, number of batches to use: {num_batches}")
    for _ in tqdm(range(num_batches)):
        batch = next(data)[0].to(dist_util.dev())
        with t.no_grad():
            for j, this_logsnr in enumerate(logsnr_grid):
                this_logsnr_broadcast = this_logsnr * t.ones(len(batch), device=batch.device)
                this_mse = t.mean(diffusion.mse([batch], this_logsnr_broadcast, mse_type='epsilon')) / num_batches
                mse_grid[j] += this_mse.cpu()
    fig = viz(logsnr_grid, mse_grid, log_eigs)
    fig.savefig('../figures/cifar_mse.png')

    logger.log("evaluating NLLs for batches...")
    samples_per_x = 50
    logger.log(f"Batch size: {args.batch_size}, samples_per_x: {samples_per_x}")
    for i, batch in enumerate(data):
        if i>10:
            break
        batch = batch[0].to(dist_util.dev())
        with t.no_grad():
            out = diffusion.nll([batch, ], logsnr_samples_per_x=samples_per_x)

        logger.log(
            f"Batch {i}: bpd={out / 32. / 32. / 3. / np.log(2.) + np.log(127.5) / np.log(2.)}")

    # TODO: Add test_nll code to use for this instead


@t.no_grad()
def mmse_x_curve(dm, x):
    """Plot estimates of MMSE_G(x, logsnr) and return fig."""
    logsnrs = t.linspace(-5, 15, 50, device=x.device)
    mmse_x = dm.mmse_g_x(x, logsnrs) * t.exp(logsnrs)
    #mses_x = dm.mse([x.repeat(len(logsnrs), 1, 1, 1)], logsnrs, mse_type='epsilon')
    mmse = dm.mmse_g(logsnrs) # * t.exp(-logsnrs)
    mses_av = 0
    for i in range(40):
        mses_av += dm.mse([x.repeat(len(logsnrs), 1, 1, 1)], logsnrs, mse_type='epsilon') / 40.
    fig, ax = plt.subplots(1)
    ax.scatter(logsnrs.cpu(), mmse_x.detach().cpu(), label='mmse_g(x,snr)')
    ax.scatter(logsnrs.cpu(), mses_av.detach().cpu(), label='MSE (average noise x 40)')
    # ax.scatter(logsnrs.cpu(), mses_x.detach().cpu(), label='MSE (x 1)')
    ax.scatter(logsnrs.cpu(), mmse.detach().cpu(), label='mmse_G(logsnr)')
    ax.set_xlabel('log SNR')
    ax.set_ylabel('$mmse_G(x, log SNR) * SNR$')
    fig.legend(loc='upper right')

    inset = fig.add_axes([0.15, 0.5, .3, .3])
    im = (0.5 + 0.5 * x).cpu().numpy().transpose((1,2,0))
    inset.imshow(im)
    plt.setp(inset, xticks=[], yticks=[])
    return fig


def create_argparser():
    defaults = dict(
        data_dir="",
        num_samples=100, # calculates nll only (=100), viz (=10) or collect (=3)
        batch_size=10,
        model_path="",
        is_viz=False,
        is_collect=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def viz(logsnrs, mses, log_eigs):
    """Default visualization for logs"""
    logsnrs = logsnrs.cpu()
    log_eigs = log_eigs.cpu()
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False)
    d = len(log_eigs)

    baseline = np.array([d / (1. + np.exp(-logsnr)) for logsnr in logsnrs])
    baseline2 = t.sigmoid(logsnrs + log_eigs.view((-1, 1))).sum(axis=0).numpy()
    ax.plot(logsnrs, baseline, label='$N(0,1)$ MMSE')
    ax.plot(logsnrs, baseline2, lw=3, label='$N(\mu,\Sigma)$ MMSE')
    ax.plot(logsnrs, mses, label='Data MSE')
    ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
    ax.set_xlabel('log SNR ($\gamma$)')
    ax.legend()

    fig.set_tight_layout(True)
    return fig


def image_movie(im_history, filename):
    im_history = im_history.transpose(0,1)  # put time-step first
    n_t = len(im_history)  # number of time-steps
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.set_title("t = {}".format(0))
    ax.set_aspect('equal')
    plt.axis('off')
    nrows = int(np.sqrt(len(im_history[0])))

    im_history = np.clip(im_history*0.5 + 0.5, 0, 1)  # imshow expects [0,1] floats
    get_grid = lambda x: tv.utils.make_grid(t.Tensor(x), nrow=nrows).cpu().numpy().transpose((1, 2, 0))
    im = ax.imshow(get_grid(im_history[0]))

    # animation function
    def animate(i):
        im.set_array(get_grid(t.Tensor(im_history[i])))
        plt.title('t = %i' % i)

    anim = animation.FuncAnimation(fig, animate, frames=n_t)
    anim.save(filename, writer=animation.FFMpegWriter(fps=10))
    plt.close(fig)


if __name__ == "__main__":
    main()
