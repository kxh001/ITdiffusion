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
import math

import sys
sys.path.append('..')  # Lame, must be a better way to do relative import
import diffusionmodel as dm
from utilsiddpm.utils import viz, plot_image
from utilsiddpm import dist_util, logger
from utilsiddpm.image_datasets import load_data, load_dataloader
from utilsiddpm.script_util import (
      model_and_diffusion_defaults,
      create_model,
      add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model = create_model(args.image_size,
        args.num_channels,
        args.num_res_blocks,
        learn_sigma=args.learn_sigma,
        class_cond=args.class_cond,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        wrapped=True)

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
        deterministic=False,
        subset=1000
    )

    # I changed the hard coded train location for my system, should come up with better solution
    train_loc = '/home/theo/Research/datasets/cifar_train'
    data_train = load_data(
        data_dir = train_loc,
        batch_size = 50000,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("calculate integral bound...")
    diffusion = dm.DiffusionModel(model)
    covariance = t.load('./scripts/cifar_covariance.pt')  # Load cached spectrum for speed
    covariance = [q.to(dist_util.dev()) for q in covariance]
    log_eigs = covariance[2]
    diffusion.dataset_info(data_train, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")

    logger.log("Generating samples")

    # cosine noise schedule
    schedule = cossche(args.diffusion_steps)

    # schedule = diffusion.generate_schedule(1000, data, plot_grid=True)
    schedule = t.load('./scheduler/schedule_cifar_500.pt')

    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        z = diffusion.sample(schedule, n_samples=args.batch_size, store_t=False, verbose=False)
        samples = ((z + 1) * 127.5).clamp(0, 255).to(t.uint8)
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.contiguous()

        gathered_samples = [t.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(r"/home/theo/Research_Results/", f"samples_{shape_str}_{args.diffusion_steps}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

def create_argparser():
    defaults = dict(
        data_dir="",
        num_samples=1000,
        batch_size=32,
        model_path="",
        is_viz=False,
        is_collect=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def cossche(num_diffusion_timesteps):
    return betas_for_alpha_bar(
        num_diffusion_timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    schedule = []
    for i in range(num_diffusion_timesteps):
        timestep = i / num_diffusion_timesteps
        cosbase = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
        schedule.append(t.logit(t.tensor(alpha_bar(timestep) / cosbase)))
    return t.flip(t.tensor(schedule), (0,))

if __name__ == "__main__":
    main()
