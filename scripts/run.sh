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
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/4.0/2.5 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:2.5}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/8.0/2.5 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:2.5}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/12/2.5 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:2.5}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/15/2.5 Trainer.max_epoch=500 Trainer.MI_params.mu=15.0 Trainer.VAT_params={eps:2.5}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/4.0/5.0 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:5.0}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/8.0/5.0 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:5.0}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/12/5.0 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:5.0}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/15/5.0 Trainer.max_epoch=500 Trainer.MI_params.mu=15.0 Trainer.VAT_params={eps:5.0}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/4.0/7.5 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:7.5}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/8.0/7.5 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:7.5}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/12/7.5 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:7.5}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/15/7.5 Trainer.max_epoch=500 Trainer.MI_params.mu=15.0 Trainer.VAT_params={eps:7.5}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/4.0/10 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:10}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/8.0/10 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:10}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/12/10 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:10}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/15/10 Trainer.max_epoch=500 Trainer.MI_params.mu=15.0 Trainer.VAT_params={eps:10}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/4.0/15 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:15}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/8.0/15 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:15}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/12/15 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:15}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/15/15 Trainer.max_epoch=500 Trainer.MI_params.mu=15.0 Trainer.VAT_params={eps:15}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/4.0/20 Trainer.max_epoch=500 Trainer.MI_params.mu=4.0 Trainer.VAT_params={eps:20}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/8.0/20 Trainer.max_epoch=500 Trainer.MI_params.mu=8.0 Trainer.VAT_params={eps:20}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/12/20 Trainer.max_epoch=500 Trainer.MI_params.mu=12.0 Trainer.VAT_params={eps:20}" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=cifar_finetunning/pretrained_resnet18/imsatvat/15/20 Trainer.max_epoch=500 Trainer.MI_params.mu=20.0 Trainer.VAT_params={eps:20}" \
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${cmd}"
#${cmd}
done
