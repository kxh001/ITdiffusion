import argparse
import os
import numpy as np
import torch.distributed as dist
import torch as t

import utilsiddpm.information_theoretic_diffusion as dm
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

    diffusion = dm.ITDiffusionModel(model)

    logger.log("creating data loader...")
    data_train_dir = '/home/theo/Research/datasets/cifar_train'
    data_test_dir = '/home/theo/Research/datasets/cifar_test'
    data_train = load_dataloader(
        data_dir=data_train_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False, 
        # subset=1000,
    )

    data_test = load_dataloader(
        data_dir=data_test_dir,
        batch_size=64,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    data_train_cov = load_data(
        data_dir = data_train_dir,
        batch_size = 12500, # for imagenet64, it could be a large number
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    logger.log("calculate integral bound...")
    covariance = t.load('./scripts/cifar_covariance.pt')# Load cached spectrum for speed
    covariance = [q.to(dist_util.dev()) for q in covariance]
    diffusion.dataset_info(data_train_cov, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")

    diffusion.fit(data_train, data_test, epochs=10, lr=1e-5, use_optimizer='adam', verbose=True)
    np.save(f"/home/theo/Research_Results/fine_tune/results.npz", diffusion.results)
    fig = viz(diffusion.logs, d=3*args.image_size**2)
    out_path = os.path.join(f"/home/theo/Research_Results/fine_tune/", f"viz.png")
    fig.savefig(out_path)
    

def create_argparser():
    defaults = dict(
        data_dir="", 
        batch_size=8, 
        model_path="", 
        is_viz=False, 
        is_collect=False,
        iddpm = False,
        wrapped = True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
