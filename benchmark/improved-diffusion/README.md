## Installation
Run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.


# Command

1. IDDPM + CIFAR10 + vlb:
> python ./scripts/image_nll.py --model_path /home/theo/Research/checkpoints/iddpm/cifar10_uncond_vlb_50M_500K.pt --data_dir /home/theo/Research/datasets/cifar_test/ --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --use_kl True --iddpm True --hugface False

2. IDDPM + CIFAR10 + hybrid:
> python ./scripts/image_nll.py --model_path /home/theo/Research/checkpoints/iddpm/cifar10_uncond_50M_500K.pt --data_dir /home/theo/Research/datasets/cifar_test/ --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --iddpm True --hugface False

3. IDDPM + Imagenet64 + vlb:
> python ./scripts/image_nll.py --model_path /home/theo/Research/checkpoints/iddpm/imagenet64_uncond_vlb_100M_1500K.pt --data_dir /home/theo/Research/datasets/Imagenet64_val/image --image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --diffusion_steps 4000 --noise_schedule cosine --use_kl True --iddpm True --hugface False

4. DDPM + CIFAR10: (for huggingface, remember to add '["sample]' in gaussian_diffusion.py [Line 261])
> python ./scripts/image_nll.py --model_path /home/theo/Research/checkpoints/ddpm_cifar10_32/diffusion_pytorch_model.bin --data_dir /home/theo/Research/datasets/cifar_test/ --diffusion_steps 4000 --noise_schedule linear --image_size 32 --iddpm False --hugface True
