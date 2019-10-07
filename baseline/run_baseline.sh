#!/usr/bin/env bash
# to verify the baseline of the project, we reproduce the IMSATVAT for MNIST and CIFAR10
## MNIST IMSATVAT
#python train_original.py Config=config_MNIST.yaml Trainer.save_dir=baseline/mnist/imsatvat Trainer.name=imsatvat
## CIFAR IMSATVAT
#python train_original_IMSAT_CIFAR.py Config=config_MNIST.yaml Trainer.save_dir=baseline/cifar/imsatvat Trainer.name=imsatvat Arch.in_channel=2048
#
## MNIST IIC
#python train_original.py Config=config_MNIST.yaml Trainer.save_dir=baseline/mnist/iicgeo Trainer.name=iicgeo \
# Trainer.head_control_params={A:2,B:1} Arch.num_sub_heads=5 Arch.name=clusternet6cTwoHead Arch.input_size=24 Arch.num_channel=1 \
#Optim.lr=0.0001 Trainer.max_epoch=1000
# CIFAR IIC using sobel
python train_original.py Config=config_CIFAR.yaml Trainer.save_dir=baseline/cifar/iicgeo/use_sobel Trainer.name=iicgeo \
 Trainer.head_control_params="{A:2,B:1}" Arch.num_sub_heads=5 Arch.name=clusternet5gTwoHead Arch.num_channel=2 \
Optim.lr=0.0001 Trainer.use_sobel=True Trainer.max_epoch=1000 Trainer.use_sobel=true
# CIFAR IIC not using sobel
python train_original.py Config=config_CIFAR.yaml Trainer.save_dir=baseline/cifar/iicgeo/use_no_sobel Trainer.name=iicgeo \
 Trainer.head_control_params="{A:2,B:1}" Arch.num_sub_heads=5 Arch.name=clusternet5gTwoHead Arch.num_channel=2 \
Optim.lr=0.0001 Trainer.use_sobel=True Trainer.max_epoch=1000 Trainer.use_sobel=false