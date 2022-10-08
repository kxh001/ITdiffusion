"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os
import numpy as np
import torch.distributed as dist
import torch as t

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
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_dataloader(
        data_dir=args.data_test_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond, 
        deterministic=True, # Generating MMSE curves with subset, better to use random subset
        # subset=100,
    )

    data_train = load_data(
        data_dir = args.data_train_dir,
        batch_size = 12500, # for imagenet64, it could be a large number
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("calculate integral bound...")
    if args.diagonal:
        diffusion.dataset_info(data_train, diagonal=args.diagonal)
    else:
        covariance = t.load('./scripts/cifar_covariance.pt' if args.image_size == 32 \
                        else './scripts/imagenet64_covariance.pt')  # Load cached spectrum for speed
        covariance = [q.to(dist_util.dev()) for q in covariance]
        diffusion.dataset_info(data_train, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")

    logger.log(f"evaluating on 100 points...")
    results, _= diffusion.test_nll(data, npoints=100, delta=1./127.5, xinterval=(-1, 1))
    np.save(f'/home/theo/Research_Results/image_nll_cifar/ddpm_vlb.npy', results)
    print(results['nll-discrete (bpd)'], results['nll-discrete-limit (bpd)'], results['nll-discrete-limit (bpd) - dequantize'])

def create_argparser():
    defaults = dict(
        data_train_dir='/home/theo/Research/datasets/cifar_train',
        data_test_dir='/home/theo/Research/datasets/cifar_test',  
        batch_size=256, 
        model_path="", 
        iddpm = True,
        wrapped = True,
        diagonal = False,
        class_cond = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
