"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os
import numpy as np
import torch.distributed as dist
import torch as t
import matplotlib.pyplot as plt
import math

import diffusionmodel as dm
from utilsiddpm.utils import viz, plot_image
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

    logger.log("creating model and diffusion...")
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    diffusion = dm.DiffusionModel(model)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        subset=100,
    )

    data_train = load_data(
        data_dir = '/home/theo/Research/datasets/cifar_train',
        batch_size = 20000,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("calculate integral bound...")
    covariance = t.load('./scripts/cifar_covariance.pt' if args.image_size == 32 \
                    else './scripts/imagenet64_covariance.pt')  # Load cached spectrum for speed
    covariance = [q.to(dist_util.dev()) for q in covariance]
    diffusion.dataset_info(data_train, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")

    logger.log("testing...")
    test(diffusion, data, 
        args.batch_size,
        args.num_samples, 
        args.diffusion_steps, 
        args.is_viz,
        args.is_collect,
    )


def test(diffusion, data, batch_size, num_samples, diffusion_steps, is_viz, is_collect):
    """
    This function includes two parts, one is calculating NLL, another is plotting interior images.
    - param model: input net model, e.g., Unet
    - param diffusion: inout diffusion model, e.g., iddpm or vdm or itdm
    - param data: batch data [B, C, ...] generator
    - param batch_size: # of samples in a batch data
    - param num_samples: # of total evaluation samples (not batch size)
    - param diffusion_steps: the timesteps or # of samples per x
    - param is_viz (bool): plot validation loss, MSE, and MSE gap 
    - param is_collect (bool): collect interior images (x, z, eps, eps_hat, eps_diff) 
    """
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        batch = (batch, ) # for extension conditioned on y
        logsnrs = t.linspace(-10, 10, 25).to('cuda')
        t_min_list = []
        gt_t_list = []
        for this_logsnr in logsnrs:
            with t.no_grad():
                batch_mse = diffusion.mse_test(batch, logsnr=this_logsnr)
                # plt.plot(batch_mse)
                # plt.show()
                gt_t = logsnr2t(logsnr=this_logsnr, total_steps=1000)
                gt_t_list.append(gt_t.cpu())
                print("Ground Truth t: ", gt_t)
                tList = t.linspace(0, 1000, 4000) 
                t_min = tList[t.argmin(t.tensor(batch_mse))]
                t_min_list.append(t_min)
                print("Min MSE t: ", t_min)
        plt.plot(t_min_list, logsnrs.cpu(), label='t_from_min_mse')
        plt.plot(gt_t_list, logsnrs.cpu(), label='t_from_logsnr2t')
        plt.xlabel('t')
        plt.ylabel('logsnr')
        plt.legend()
        plt.show()
        num_complete += dist.get_world_size() * batch[0].shape[0]
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", 
        num_samples=1, # calculates nll only (=100), viz (=10) or collect (=3)
        batch_size=1, 
        model_path="", 
        is_viz=False, 
        is_collect=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def logsnr2t(logsnr, s=0.008, total_steps=1000):
    timestep = total_steps * (t.acos(t.cos(t.tensor(s / (s + 1) * math.pi / 2)) * t.sqrt(t.sigmoid(logsnr))) * 2 * (1 + s) / math.pi - s)
    return timestep

if __name__ == "__main__":
    main()
