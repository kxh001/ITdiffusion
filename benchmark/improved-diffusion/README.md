# Benchmark -- Diffusion Model with Variational Bounds

- We added one more flag 'cont_density' in the original IDDPM codes, which could be found in 
[losses.py](https://github.com/kxh001/ITdiffusion/blob/main/benchmark/improved-diffusion/improved_diffusion/losses.py). 
- If 'cont_density=True', it calculates the **continuous** NLL for diffusion models using variational bounds (the 2nd column in Table 1),
otherwise, it will calculate the **discrete** NLL (the 1st column in Table 2). 
- The way to calculate **continuous** NLL for discrete estimator (the 1st column of Table 1) is assuming uniform density in each bin. Thus, the calculation should be:
$\text{discrete NLL} - \frac{log(127.5)}{log(2)} $
- We don't include the results of benchmark here, you could run command lines in [Run](#3) to get all the results. 

## Installation
Navigate to './benchmark/improved-diffusion' as working directory in the Linux terminal or Anaconda Powershell Prompt, then run the command:

```
pip install -e .
```

This would install the 'improved-diffusion' python package that scripts depend on. 

<h2 id='3'>Run</h3>
1. IDDPM + CIFAR10 + vlb:
```
python ./scripts/image_nll.py
--model_path XXX/cifar10_uncond_vlb_50M_500K.pt --data_dir XXX/cifar_test 
--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 
--diffusion_steps 4000 --noise_schedule cosine --use_kl True --iddpm True --cont_density True
```

2. DDPM + CIFAR10:
```
python ./scripts/image_nll.py 
--data_dir XXX/cifar_test
--image_size 32
--diffusion_steps 4000 --noise_schedule linear --iddpm False --cont_density True
```
