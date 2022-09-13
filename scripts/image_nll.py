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
    model, diffusion = create_model_and_diffusion(
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
                    else '/home/theo/Research/datasets/Imagenet64_val/Imagenet64_val',
        batch_size = 20000,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("calculate integral bound...")
    covariance = t.load('./scripts/cifar_covariance.pt' if args.image_size == 32 \
                    else './scripts/imagenet64_covariance.pt')  # Load cached spectrum for speed
    covariance = [q.to(dist_util.dev()) for q in covariance]
    log_eigs = covariance[2]
    diffusion.dataset_info(data_train, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")

    logger.log("evaluating...")
    # run_nll_evaluation(model, diffusion, data, 
    #     args.image_size,
    #     args.batch_size,
    #     args.num_samples, 
    #     args.diffusion_steps, 
    #     args.is_viz,
    #     args.is_collect,
    # )
    results = diffusion.test_nll(data, npoints=1000, delta=1./127.5, xrange=(-1,1))
    print(results)


def run_nll_evaluation(model, diffusion, data, image_size, batch_size, num_samples, diffusion_steps, is_viz, is_collect):
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
    all_nll = []
    all_metrics = {"nll": []}
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        batch = (batch, ) # for extension conditioned on y
        batch_metric = diffusion.calc_nll(model, batch, diffusion_steps, num_samples/batch_size, is_viz, is_collect)
        diffusion.log_function(batch_metric["nll"], diffusion.mse_grid)

        for key, term_list in all_metrics.items():
            terms = batch_metric[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_nll = batch_metric["nll"]
        total_nll = total_nll.mean() / dist.get_world_size()
        dist.all_reduce(total_nll)
        all_nll.append(total_nll.item())
        num_complete += dist.get_world_size() * batch[0].shape[0]

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_nll)/image_size/image_size/3./np.log(2.) + np.log(127.5)/np.log(2.)}")

    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))
    if is_viz: 
        print("Ploting the mse curve...")
        fig = viz(diffusion.logs, d=3*image_size**2)
        out_path = os.path.join(logger.get_dir(), f"viz.png")
        fig.savefig(out_path)
    if is_collect:
        print("Ploting the interior images when logsnr = [-2, 2, 8, 10]...")
        plot_image(diffusion.image_collection, num_samples, diffusion.logs['logsnr_grid'].numpy())
    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", 
        num_samples=100, # calculates nll only (=100), viz (=10) or collect (=3)
        batch_size=10, 
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
