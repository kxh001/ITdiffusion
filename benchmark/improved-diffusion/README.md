## Installation
Run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.


# Command for AWS (go into the "~/ITdiffusion-main/benchmark/improved-diffusion" directory)

1. IDDPM + CIFAR10 + vlb:
> python ./scripts/image_nll.py --model_path /home/ubuntu/ITdiffusion-main/models/cifar10_uncond_vlb_50M_500K.pt --data_dir /home/ubuntu/ITdiffusion-main/data/random100/cifar_test --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --use_kl True --iddpm True --hugface False --cont_density True

2. IDDPM + CIFAR10 + hybrid:
> python ./scripts/image_nll.py --model_path /home/ubuntu/ITdiffusion-main/models/cifar10_uncond_50M_500K.pt --data_dir /home/ubuntu/ITdiffusion-main/data/random100/cifar_test/ --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --iddpm True --hugface False --cont_density True

3. IDDPM + Imagenet64 + vlb:
> python ./scripts/image_nll.py --model_path /home/ubuntu/ITdiffusion-main/models/imagenet64_uncond_vlb_100M_1500K.pt --data_dir /home/ubuntu/ITdiffusion-main/data/random100/Imagenet64_val --image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --diffusion_steps 4000 --noise_schedule cosine --use_kl True --iddpm True --hugface False --cont_density True

4. DDPM + CIFAR10:
> python ./scripts/image_nll.py --model_path /home/ubuntu/ITdiffusion-main/models/ddpm_cifar10_32/diffusion_pytorch_model.bin --data_dir /home/ubuntu/ITdiffusion-main/data/random100/cifar_test/ --diffusion_steps 4000 --noise_schedule linear --image_size 32 --iddpm False --hugface True --cont_density True
