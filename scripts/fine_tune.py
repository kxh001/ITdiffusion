import argparse
import os
import numpy as np
import torch.distributed as dist
import torch as t

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

    logger.log("creating data loader...")
    data_train = load_dataloader(
        data_dir=args.data_train_dir,
        batch_size=args.train_batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False, 
    )

    data_test = load_dataloader(
        data_dir=args.data_test_dir,
        batch_size=args.test_batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    data_train_cov = load_data(
        data_dir = args.data_train_dir,
        batch_size = 12500, # for cifar10, it could be a small number
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    logger.log("calculate integral bound...")
    if args.diagonal:
        diffusion.dataset_info(data_train_cov, diagonal=args.diagonal)
    else:
        covariance = t.load('./scripts/cifar_covariance.pt')  # Load cached spectrum for speed
        covariance = [q.to(dist_util.dev()) for q in covariance]
        diffusion.dataset_info(data_train_cov, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")


    logger.log("fine tune model...")
    if args.is_test:
        diffusion.fit(data_train, data_test, epochs=args.epoch, lr=args.lr, use_optimizer='adam', verbose=True, iddpm=args.iddpm)
    else:
        diffusion.fit(data_train, epochs=args.epoch, lr=args.lr, use_optimizer='adam', iddpm=args.iddpm)

    logger.log("save results...")
    if args.iddpm:
        out_path = f"/media/theo/Data/checkpoints/iid_sampler/iddpm"
    else:
        out_path = f"/media/theo/Data/checkpoints/iid_sampler/ddpm"
    np.save(os.path.join(out_path,"train_loss_all.npy"), diffusion.logs['train loss'])
    np.save(os.path.join(out_path,"test_loss_all.npy"), diffusion.logs['val loss'])

def create_argparser():
    defaults = dict(
        data_train_dir='/home/theo/Research/datasets/cifar_train',
        data_test_dir='/home/theo/Research/datasets/cifar_test',  
        train_batch_size=128, 
        test_batch_size=256,
        model_path="/home/theo/Research/checkpoints/iddpm/cifar10_uncond_vlb_50M_500K.pt", 
        covar_path='/home/theo/Research/ITD/diffusion/covariance/cifar_covariance.pt', 
        lr =2e-4,
        epoch=10,
        iddpm = True,
        wrapped = True,
        class_cond = False,
        diagonal = False,
        is_test = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
