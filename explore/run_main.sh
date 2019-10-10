#!/usr/bin/env bash
set -e
max_epoch=2000
batch_size=100

# single head comparison
python main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Arch.name=clusternet6cTwoHead_sn Trainer.save_dir=explore/MNIST/singlehead/sn Trainer.max_epoch=${max_epoch} DataLoader.batch_size=${batch_size}
python main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Arch.name=clusternet6cTwoHead Trainer.save_dir=explore/MNIST/singlehead/non_sn Trainer.max_epoch=${max_epoch} DataLoader.batch_size=${batch_size}

python main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeo Arch.name=clusternet5gTwoHead_sn Trainer.save_dir=explore/CIFAR/singlehead/sn Trainer.max_epoch=${max_epoch} DataLoader.batch_size=${batch_size}
python main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeo Arch.name=clusternet5gTwoHead Trainer.save_dir=explore/CIFAR/singlehead/non_sn Trainer.max_epoch=${max_epoch} DataLoader.batch_size=${batch_size}
# multihead comparison
#python main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Arch.name=clusternet6cTwoHead_sn Trainer.save_dir=explore/MNIST/multihead/sn Trainer.max_epoch=${max_epoch} \
#Arch.num_sub_heads=5 Trainer.head_control_params.A=2 Trainer.head_control_params.B=1 DataLoader.batch_size=${batch_size}
#python main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Arch.name=clusternet6cTwoHead Trainer.save_dir=explore/MNIST/multihead/non_sn Trainer.max_epoch=${max_epoch} \
#Arch.num_sub_heads=5 Trainer.head_control_params.A=2 Trainer.head_control_params.B=1 DataLoader.batch_size=${batch_size}
#
#python main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeo Arch.name=clusternet5gTwoHead_sn Trainer.save_dir=explore/CIFAR/multihead/sn Trainer.max_epoch=${max_epoch} \
#Arch.num_sub_heads=5 Trainer.head_control_params.A=2 Trainer.head_control_params.B=1 DataLoader.batch_size=${batch_size}
#python main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeo Arch.name=clusternet5gTwoHead Trainer.save_dir=explore/CIFAR/multihead/non_sn Trainer.max_epoch=${max_epoch} \
#Arch.num_sub_heads=5 Trainer.head_control_params.A=2 Trainer.head_control_params.B=1 DataLoader.batch_size=${batch_size}