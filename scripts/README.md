- The pre-trianed IDDPM model could be downloaded at [checkpoint](https://github.com/openai/improved-diffusion).

- The pre-trained DDPM model is [here](https://drive.google.com/drive/folders/1G1nFv6AML_8zeElxMECkYfJnmihVcR86?usp=sharing) 

- The dataset and preprocessing could be found at [dataset_generation](https://github.com/openai/improved-diffusion/tree/main/datasets).

- The following commands are used to run image_nll.py:

1. IDDPM + CIFAR10 + vlb:
>python ./scripts/image_nll.py --model_path /home/theo/Research/checkpoints/iddpm/cifar10_uncond_vlb_50M_500K.pt --data_dir /home/theo/Research/datasets/cifar_test/ --diffusion_steps 4000 --iddpm True --wrapped True --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 

2. DDPM + CIFAR10:
>python ./scripts/image_nll.py --model_path /home/theo/Research/checkpoints/ddpm_cifar10_32/diffusion_pytorch_model.bin --data_dir /home/theo/Research/datasets/cifar_test/ --diffusion_steps 4000 --iddpm False --wrapped True --image_size 32

- The following commands are used to run fine_tune.py:
1. IDDPM + CIFAR10 + vlb:
> python ./scripts/fine_tune.py 
--data_train_dir /home/theo/Research/datasets/cifar_train --data_test_dir /home/theo/Research/datasets/cifar_test
--model_path /home/theo/Research/checkpoints/iddpm/cifar10_uncond_vlb_50M_500K.pt
--covar_path /home/theo/Research/ITD/diffusion/covariance/cifar_covariance.pt
--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --iddpm True --wrapped True
--train_batch_size 128 --test_batch_size 256 --lr 1e-4 --epoch 5

2. DDPM + CIFAR10:
> python ./scripts/fine_tune.py --data_train_dir /home/theo/Research/datasets/cifar_train --data_test_dir /home/theo/Research/datasets/cifar_test --model_path /home/theo/Research/checkpoints/ddpm_cifar10_32/diffusion_pytorch_model.bin --covar_path /home/theo/Research/ITD/diffusion/covariance/cifar_covariance.pt --iddpm False --wrapped True --image_size 32 --train_batch_size 128 --test_batch_size 256 --lr 2e-4 --epoch 5
  

For cifar_tests.py GV has been using this command from the scripts folder:
> python cifar_tests.py --model_path /home/gregv/diffusion/models/cifar10_uncond_50M_500K.pt --data_dir /home/gregv/diffusion/data/cifar_test --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000
> 
> python cifar_tests.py --model_path /home/gregv/ITdiffusion/models/cifar10_uncond_50M_500K.pt --data_dir /home/gregv/ITdiffusion/data/cifar_test --image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --diffusion_steps 4000 --batch_size 100 --iddpm True --wrapped True

Note that for sampling, the first step is to generate 
a schedule from the MMSE curve. 
The line:
> schedule = diffusion.generate_schedule(500, data)

or 

> schedule_info = diffusion.generate_schedule(500, data, info_init=True)

does this, using data and a schedule of 500 steps. It is a bit slow.
I recommend using a small number of points in "data" for estimating this. (It shouldn't be very sensitive. I used 200 samples.)
Then I recommend saving it and loading it to speed up sampling experiments.

I got the best looking movies using these two combinations. 
> z = diffusion.sample(schedule, n_samples=9, store_t=5, precondition=False, temp=1., info_init=False, m=2)
z = diffusion.sample(schedule_info, n_samples=9, store_t=5, precondition=False, temp=1., info_init=True, m=2)

"precondition" I couldn't get to work at all - it's based on the principle from Neal's HMC paper that we should try to approximately diagonalize 
the data (using the covariance matrix which we have calculated). Maybe it requires some different learning rate or I made a mistake somewhere. 

info_init says whether we start with samples from N(0,1) or N(mu, Sigma), matching covariance of the data.
Both seem reasonable. 

Temperature in the range 0,1 controls the noise added in Langevin dynamics. 1 is proper, but sometimes I thought images 
looked a little better with 0.8 or 0.9. 

The step size for Langevin dynamics is hard-coded. It is based on the logic of mimicking the 
previous literature that we should estimate the mean p(z0|z1) where z0 is a sample at the target SNR
and z1 is the current sample at a given SNR. 
HOWEVER, I found that a mystery factor, m=2, improves results dramatically (while m=1 would match previous work most closely).
This "m" means that we are actually targeting an SNR twice as large as the next SNR in our schedule. 
