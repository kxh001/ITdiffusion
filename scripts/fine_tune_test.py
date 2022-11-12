"""
Approximate the bits/dimension for an image model.
"""

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

    # model_name_list = ["ddpm_model_epoch1.pt","ddpm_model_epoch2.pt","ddpm_model_epoch3.pt","ddpm_model_epoch4.pt","ddpm_model_epoch5.pt",
    #                 "ddpm_model_epoch6.pt","ddpm_model_epoch7.pt","ddpm_model_epoch8.pt","ddpm_model_epoch9.pt","ddpm_model_epoch10.pt"]
    
    # model_name_list = ["DDPM_epoch1.pt","DDPM_epoch2.pt","DDPM_epoch3.pt","DDPM_epoch4.pt","DDPM_epoch5.pt",
    #                     "DDPM_epoch6.pt","DDPM_epoch7.pt","DDPM_epoch8.pt","DDPM_epoch9.pt","DDPM_epoch10.pt"]
    
    # model_name_list = ["model_epoch1.pt","model_epoch2.pt","model_epoch3.pt","model_epoch4.pt","model_epoch5.pt","model_epoch6.pt",
                        # "model_epoch7.pt","model_epoch8.pt","model_epoch9.pt","model_epoch10.pt"]
    model_name_list = ["model_epoch0.pt"]
    if args.iddpm:    
      model_path = "D:/checkpoints/fine_tune_soft/iddpm/"
    else:
      model_path = "D:/checkpoints/fine_tune_soft/ddpm/"
    for mname in model_name_list:
        # mpath = os.path.join(model_path, mname)
        mpath = 'C:/Users/72809/Desktop/Research/checkpoints/iddpm/cifar10_uncond_50M_500K.pt'
        # mpath = 'C:/Users/72809/Desktop/Research/checkpoints/iddpm/cifar10_uncond_vlb_50M_500K.pt'
        # mpath = 'C:/Users/72809/Desktop/Research/checkpoints/ddpm_cifar10_32/diffusion_pytorch_model.bin'
        model.load_state_dict(
            dist_util.load_state_dict(mpath, map_location="cpu"), strict=True
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
            covariance = t.load('./scripts/cifar_covariance.pt')  # Load cached spectrum for speed
            covariance = [q.to(dist_util.dev()) for q in covariance]
            diffusion.dataset_info(data_train, covariance_spectrum=covariance)
        logger.log(f"loc_logsnr:{diffusion.loc_logsnr}, scale_logsnr:{diffusion.scale_logsnr}")

        logger.log(f"evaluating on {args.npoints} points...")
        results, val_loss = diffusion.test_nll(data, npoints=args.npoints, delta=1./127.5, xinterval=(-1, 1), soft=False)
        print(results)
        results2, val_loss = diffusion.test_nll(data, npoints=args.npoints, delta=1./127.5, xinterval=(-1, 1), soft=True)
        print(results2)
        np.save(f'./results/fine_tune/iddpm_hybrid/toy/fine_tune_test/results_epoch{mname[11:-3]}_hard.npy', results)
        np.save(f'./results/fine_tune/iddpm_hybrid/toy/fine_tune_test/results_epoch{mname[11:-3]}_soft.npy', results2)
        # import IPython; IPython.embed()

        # if args.iddpm:
        #     np.save(f'./results/fine_tune/iddpm_hybrid/results_epoch{mname[11:-3]}.npy', [results, val_loss])
        # else:
        #     np.save(f'./results/debug/soft_discretization/ddpm/results_epoch{mname[11:-3]}.npy', [results, val_loss])
        logger.log('epoch: {}\t val loss: {:0.4f}'.format(mname[11:-3], val_loss))
        logger.log('nll (nats): {:0.4f}\t nll (bpd): {:0.4f}\t nll-discrete (bpd): {:0.4f}\t nll-discrete-limit (bpd) - dequantize:{:0.4f}'
                  .format(results['nll (nats)'],
                          results['nll (bpd)'],
                          results['nll-discrete (bpd)'], 
                          results['nll-discrete-limit (bpd) - dequantize']))

def create_argparser():
    defaults = dict(
        data_train_dir=r'C:/Users/72809/Desktop/Research/datasets/cifar_train',
        data_test_dir=r'C:/Users/72809/Desktop/Research/datasets/random100/cifar_test', 
        batch_size=256, 
        iddpm = True,
        wrapped = True,
        diagonal = False,
        npoints = 100,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
