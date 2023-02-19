# Information-Theoretic Diffusion (ITD)

Code for the paper [Information-Theoretic Diffusion](https://openreview.net/forum?id=UvmDCdSPDOW), published at ICLR 2023.

We introduce a new mathematical foundation for diffusion models inspired by classic results in information theory, which yields a unified objective for modeling either continuous or discrete data and provides justification for ensembling of diffusion models.

$$ \log p(x) = - \frac{1}{2} \int_{0}^{\infty} \text{mmse}(x, \gamma) d\gamma + \text{const} \qquad \text{where} \quad \text{mmse} = \min_{\hat{x}} E_{p(z_{\gamma}|x)}\big[ \| x - \hat{x}(z_{\gamma}, \gamma) \|^2 \big] $$

where $\gamma$ is the signal-to-noise ratio.   For discrete likelihood estimation, we visualize the curve of denoising errors by log SNR (left) and show that existing models can be improved using fine-tuning inspired by the ITD loss or by ensembling different models at different SNRs (right).

![Discrete Results](/results/figs/discrete_fig_table.png)
 
<!-- Initial commit for improved and generalized applications of diffusion models based on an information-theoretic formulation.  -->

For better understanding, a [simplified demonstration](https://github.com/gregversteeg/InfoDiffusionSimple) is also provided with vivid visualization and examples.


# Usage
## Installation
Clone this repository and navigate to './ITdiffusion' as working directory in the Linux terminal or Anaconda Powershell Prompt, then run the command:

```
pip install -e .
```

This would install the 'itdiffusion' python package that scripts depend on.

(<span style="color:red">**Note**</span>: If you meet troubles when installing the 'mpi4py' library, please refer [here](https://pypi.org/project/mpi4py/).)


## Utilities
Folder 'utilsitd' includes the utilities from [improved-diffusion](https://github.com/openai/improved-diffusion) and our ITdiffusion model, and then use scripts to output results we desire. 

## Preparing Data
We use CIFAR-10 dataset in our paper. The dataset preprocessing code is provided by [dataset generation](https://github.com/openai/improved-diffusion/tree/main/datasets).
For convenience, we include it in [cifar10.py](https://github.com/kxh001/ITdiffusion/blob/main/datasets/cifar10.py). You could run it directly to get processed dataset.

## Fine-tuning
#### The following commands are used to run 'fine_tune.py':
1. IDDPM + CIFAR10 + vlb:
```
python ./scripts/fine_tune.py 
--data_train_dir XXX/cifar_train 
--model_path XXX/iddpm/cifar10_uncond_vlb_50M_500K.pt 
--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 
--iddpm True --train_batch_size 32 --lr 2.5e-5 --epoch 10
```
2. DDPM + CIFAR10:
```
python ./scripts/fine_tune.py 
--data_train_dir XXX/cifar_train
--image_size 32
--iddpm False --train_batch_size 64 --lr 1e-4 --epoch 10
```

#### For evaluation, run 'test.py' directly:
1. IDDPM + CIFAR10 + vlb:
```
python ./scripts/test.py 
--data_train_dir XXX/cifar_train --data_test_dir XXX/cifar_test
--model_path ../checkpoints/iddpm/model_epoch10.pt 
--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 
--iddpm True --test_batch_size 256 --lr 2.5e-5 --npoints 1000 --soft True
```
2. DDPM + CIFAR10:
```
python ./scripts/test.py 
--data_train_dir XXX/cifar_train --data_test_dir XXX/cifar_test
--image_size 32
--iddpm False --test_batch_size 256 --lr 1e-4 --npoints 1000 --soft True
```

## Models
- The pre-trained IDDPM model could be downloaded [here](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_vlb_50M_500K.pt).

- We use pre-trained DDPM model from Huggingface via [diffusers](https://github.com/huggingface/diffusers) library.

- Fined-tuned models could be found [here](https://github.com/kxh001/ITdiffusion/tree/main/checkpoints).


## Results
- Run ```python ./script/plot_results.py``` to get figures and tables in the paper.

- To make it clearer, we summarized **discrete** $\rightleftharpoons$ **continuous** methods used in our experiments in the following tables:
<style type="text/css">
.tg  {border-collapse:collapse;border-color:#93a1a1;border-spacing:0;}
.tg td{background-color:#fdf6e3;border-color:#93a1a1;border-style:solid;border-width:0px;color:#002b36;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#657b83;border-color:#93a1a1;border-style:solid;border-width:0px;color:#fdf6e3;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-n6w4{background-color:#FDF6E3;border-color:inherit;color:#002B36;text-align:left;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-lboi"></th>
    <th class="tg-c3ow">Continuous NLL</th>
    <th class="tg-0pky">Column # in Table 1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Discrete Estimator <br>(benchmarks)</td>
    <td class="tg-0pky">assume uniform density in each bin</td>
    <td class="tg-c3ow">1st</td>
  </tr>
  <tr>
    <td class="tg-n6w4"><span style="color:#002B36;background-color:#FDF6E3">interpret the last denosing step as </span><br><span style="color:#002B36;background-color:#FDF6E3">providing a Gaussian distribution over $\bm x$</span></td>
    <td class="tg-9wq8">2nd</td>
  </tr>
</tbody>
</table>

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#93a1a1;border-spacing:0;}
.tg td{background-color:#fdf6e3;border-color:#93a1a1;border-style:solid;border-width:0px;color:#002b36;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#657b83;border-color:#93a1a1;border-style:solid;border-width:0px;color:#fdf6e3;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-lboi"></th>
    <th class="tg-c3ow">Discrete NLL</th>
    <th class="tg-0pky">Column # in Table 2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Continuous Estimator <br>(ours)</td>
    <td class="tg-0pky">soft discretization</td>
    <td class="tg-c3ow">2nd</td>
  </tr>
  <tr>
    <td class="tg-0pky">uniform dequantization</td>
    <td class="tg-c3ow">3rd</td>
  </tr>
</tbody>
</table>



(<span style="color:red">**Note**</span>: For benchmark results with variational bounds, please read the [README.md](https://github.com/kxh001/ITdiffusion/blob/main/benchmark/improved-diffusion).)


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
- See our [simplified demonstration](https://github.com/gregversteeg/InfoDiffusionSimple) for better understanding.

