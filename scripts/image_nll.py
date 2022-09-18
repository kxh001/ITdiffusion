"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os
import numpy as np
import torch.distributed as dist
import torch as t

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
    data = load_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        subset=100
    )

    data_train = load_data(
        data_dir = '/home/theo/Research/datasets/cifar_train' if args.image_size == 32 \
                    else '/home/theo/Research/datasets/Imagenet64_train/image',
        batch_size = 12500,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("calculate integral bound...")
    covariance = t.load('./scripts/cifar_covariance.pt' if args.image_size == 32 \
                    else './scripts/imagenet64_covariance.pt')  # Load cached spectrum for speed
    covariance = [q.to(dist_util.dev()) for q in covariance]
    diffusion.dataset_info(data_train, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")

    logger.log("evaluating...")
    results = diffusion.test_nll(data, npoints=100, delta=1./127.5, xinterval=(-1, 1))
    # np.save('/home/theo/Research_Results/image_nll_test/iddpm_vlb.npy', results)
    print(results['nll-discrete (bpd)'], results['nll-discrete-limit (bpd)'])

def create_argparser():
    defaults = dict(
        data_dir="", 
        num_samples=100, # calculates nll only (=100), viz (=10) or collect (=3)
        batch_size=10, 
        model_path="", 
        is_viz=False, 
        is_collect=False,
        iddpm = True,
        wrapped = True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
