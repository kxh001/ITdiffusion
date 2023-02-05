# Information-Theoretic diffusion

Code for the paper [Information-Theoretic Diffusion](https://openreview.net/forum?id=UvmDCdSPDOW), published at ICLR 2023.

We introduce a new mathematical foundation for diffusion models inspired by classic results in information theory, which yields a unified objective for modeling either continuous or discrete data and provides justification for ensembling of diffusion models.
Our bounds take the form
$$ \log P(\vx) = - \frac{1}{2} \int_{0}^{\infty} \text{mmse}(x, \gamma) d\gamma $$


```
@inproceedings{
kong2023informationtheoretic,
title={Information-Theoretic Diffusion},
author={Xianghao Kong and Rob Brekelmans and Greg Ver Steeg},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=UvmDCdSPDOW} }
```
 
<!-- Initial commit for improved and generalized applications of diffusion models based on an information-theoretic formulation.  -->


# Usage
## Installation
Clone this repository and navigate to ./ITdiffusion as working directory, then run:

```
pip install -e .
```

This should instabll the 'itdiffusion' python package that scripts depend on. 

## Information Theoretic Diffusion
Folder 'utilsitd' includes the utilities from [improved-diffusion](https://github.com/openai/improved-diffusion) and our ITdiffusion model named 'diffusionmodel.py', and use scripts to output results we desire. 


## Preparing Data
We only use CIFAR-10 in our project. The dataset and preprocessing could be found at [dataset_generation](https://github.com/openai/improved-diffusion/tree/main/datasets).

If you would like to create your own dataset, please refer to [instructions](https://github.com/openai/improved-diffusion).

## Fine-tuning
The following commands are used to run ./scripts/fine_tune.py:
1. IDDPM + CIFAR10 + vlb:
```
python ./scripts/fine_tune.py 
--data_train_dir XXX/cifar_train --data_test_dir XXX/cifar_test
--model_path XXX/iddpm/cifar10_uncond_vlb_50M_500K.pt 
--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 
--iddpm True --wrapped True --train_batch_size 32 --test_batch_size 256 --lr 2.5e-5 --epoch 10 --test True
```
2. DDPM + CIFAR10:
```
python ./scripts/fine_tune.py 
--data_train_dir XXX/cifar_train --data_test_dir XXX/cifar_test
--model_path XXX/ddpm_cifar10_32/diffusion_pytorch_model.bin --model_config_path XXX/ddpm_cifar10_32/config.json 
--image_size 32
--iddpm False --wrapped True --train_batch_size 64 --test_batch_size 256 --lr 1e-4 --epoch 10 --test True
```

## Models
- The pre-trianed IDDPM model could be downloaded [here](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_vlb_50M_500K.pt). 

- The pre-trained DDPM model is stored [here](https://drive.google.com/drive/folders/1G1nFv6AML_8zeElxMECkYfJnmihVcR86?usp=sharing). For DDPM, it attaches with a config.json file and has the same interface of Huggingface. 


## Results
- MSE curves: [cont_density](./results/figs/cont_density.pdf), [disc_density](./results/figs/disc_density.pdf).

