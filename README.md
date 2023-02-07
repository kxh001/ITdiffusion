# Information-Theoretic Diffusion (ITD)

Code for the paper [Information-Theoretic Diffusion](https://openreview.net/forum?id=UvmDCdSPDOW), published at ICLR 2023.

We introduce a new mathematical foundation for diffusion models inspired by classic results in information theory, which yields a unified objective for modeling either continuous or discrete data and provides justification for ensembling of diffusion models.

$$ \log p(x) = - \frac{1}{2} \int_{0}^{\infty} \text{mmse}(x, \gamma) d\gamma + \text{const} \qquad \text{where} \quad \text{mmse} = \min_{\hat{x}} E_{p(z_{\gamma}|x)}\big[ \| x - \hat{x}(z_{\gamma}, \gamma) \|^2 \big] $$

where $\gamma$ is the signal-to-noise ratio.   For discrete likelihood estimation, we visualize the curve of denoising errors by log SNR (left) and show that existing models can be improved using fine-tuning inspired by the ITD loss or by ensembling different models at different SNRs (right).

![Discrete Results](/results/figs/discrete_fig_table.png)
 
<!-- Initial commit for improved and generalized applications of diffusion models based on an information-theoretic formulation.  -->


# Usage
## Installation
Clone this repository and navigate to './ITdiffusion' as working directory in the Linux terminal or Anaconda Powershell Prompt, then run the command:

```
pip install -e .
```

This would install the 'itdiffusion' python package that scripts depend on. 

## Information Theoretic Diffusion
Folder 'utilsitd' includes the utilities from [improved-diffusion](https://github.com/openai/improved-diffusion) and our ITdiffusion model, and then use scripts to output results we desire. 

## Preparing Data
We use CIFAR-10 dataset in our paper. The dataset preprocessing code is provided by [dataset_generation](https://github.com/openai/improved-diffusion/tree/main/datasets).
For convenience, we include it in './datasets/cifar10.py'. You could run it directly to get processed dataset.

## Fine-tuning
The following commands are used to run './scripts/fine_tune.py':
1. IDDPM + CIFAR10 + vlb:
```
python ./scripts/fine_tune.py 
--data_train_dir XXX/cifar_train --data_test_dir XXX/cifar_test
--model_path XXX/iddpm/cifar10_uncond_vlb_50M_500K.pt 
--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 
--iddpm True --wrapped True --train_batch_size 32 --test_batch_size 256 --lr 2.5e-5 --epoch 10 --test True --soft False
```
2. DDPM + CIFAR10:
```
python ./scripts/fine_tune.py 
--data_train_dir XXX/cifar_train --data_test_dir XXX/cifar_test
--image_size 32
--iddpm False --wrapped True --train_batch_size 64 --test_batch_size 256 --lr 1e-4 --epoch 10 --test True --soft False
```

## Models
- The pre-trained IDDPM model could be downloaded [here](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_vlb_50M_500K.pt).

- We use pre-trained DDPM model from Huggingface via '[diffusers](https://github.com/huggingface/diffusers)' library.


## Results
Run './script/plot_results.py' to get all figures and tables in the paper. 

## BibTeX
```
@inproceedings{
kong2023informationtheoretic,
title={Information-Theoretic Diffusion},
author={Xianghao Kong and Rob Brekelmans and Greg {Ver Steeg}},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=UvmDCdSPDOW} }
```

## References
- Alex Nichol's [implement of IDDPM](https://github.com/openai/improved-diffusion).
- HuggingFace's [diffusers](https://github.com/huggingface/diffusers) library.
