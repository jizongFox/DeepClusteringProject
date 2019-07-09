#!/usr/bin/env bash
CURRENT_PATH=$(pwd)
PROJECT_PATH="$(dirname "${CURRENT_PATH}")"
WRAPPER_PATH=${PROJECT_PATH}"/library/deep-clustering-toolbox/deepclustering/utils/CC_wrapper.sh"
echo "The project path: ${PROJECT_PATH}"
echo "The current path: ${CURRENT_PATH}"
echo "The wrapper path: ${WRAPPER_PATH}"
cd ${PROJECT_PATH}
source $WRAPPER_PATH
cd ${PROJECT_PATH}
set -e
time=8

declare -a StringArray=(
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeo Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/iicgeo/sobel/_ Trainer.max_epoch=500 Arch.num_channel=2 Trainer.use_sobel=true" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeo Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/iicgeo/no_sobel/_ Trainer.max_epoch=500 " \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/4.0/0.25 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:0.25}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/8.0/0.25 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:0.25}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/12/0.25 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:0.25}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/15/0.25 Trainer.max_epoch=500 Trainer.MI_params.mu=15.0 Trainer.VAT_params={eps:0.25}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/4.0/1.0 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:1.0}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/8.0/1.0 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:1.0}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/12/1.0 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:1.0}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/15/1.0 Trainer.max_epoch=500 Trainer.MI_params.mu=15.0 Trainer.VAT_params={eps:1.0}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/4.0/0.1 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:0.1}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/8.0/0.1 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:0.1}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/12/0.1 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:0.1}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_reduced_aug_for_eps/imsatvat/15/0.1 Trainer.max_epoch=500 Trainer.MI_params.mu=15.0 Trainer.VAT_params={eps:0.1}" \
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${cmd}"
#${cmd}
done
