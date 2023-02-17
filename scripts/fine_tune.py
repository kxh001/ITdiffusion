import argparse
import os
import numpy as np
import torch as t

from utilsitd import logger
from utilsitd.image_datasets import load_data, load_dataloader
from utilsitd.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.iddpm:
        model.load_state_dict(
            t.load(args.model_path, map_location="cpu")
        )
    dev = "cuda" if t.cuda.is_available() else "cpu"
    model.to(dev)
    logger.log(f"Using {dev} for DiffusionModel")

    logger.log("creating data loader...")
    data_train = load_dataloader(
        data_dir=args.data_train_dir,
        batch_size=args.train_batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False, 
    )

    data_train_cov = load_data(
        data_dir = args.data_train_dir,
        batch_size = 12500,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("calculate integral bound...")
    if args.diagonal:
        diffusion.dataset_info(data_train_cov, diagonal=args.diagonal)
    else:
        covariance = t.load('./covariance/cifar_covariance.pt')  # Load cached spectrum for speed
        covariance = [q.to(dev) for q in covariance]
        diffusion.dataset_info(data_train_cov, covariance_spectrum=covariance)
    logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")


    logger.log("fine tune model...")
    diffusion.fit(data_train, epochs=args.epoch, lr=args.lr, use_optimizer='adam')

    logger.log("save results...")
    out_path = os.path.join(logger.get_dir(), f"train_loss_all.npy")
    np.save(out_path, diffusion.logs['train loss'])

def create_argparser():
    defaults = dict(
        data_train_dir="",
        data_test_dir="",  
        train_batch_size=128,
        model_path="",
        lr=2.5e-5,
        epoch=10,
        iddpm=True, # 'Ture' if using iddpm, 'False' if using ddpm
        diagonal = False, # 'True' if data size is too large to compute covariance matrix from limited data, else 'False'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
