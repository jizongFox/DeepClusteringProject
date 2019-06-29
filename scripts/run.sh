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

time=12

declare -a StringArray=(
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=mnist_1/iicgeo Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=mnist_2/iicgeo Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=mnist_3/iicgeo Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=mnist_4/iicgeo Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=mnist_5/iicgeo Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=mnist_1/imsat Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=mnist_2/imsat Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=mnist_3/imsat Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=mnist_4/imsat Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=mnist_5/imsat Trainer.max_epoch=100" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_1/imsatvat_0.25 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=0.25}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_2/imsatvat_0.25 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=0.25}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_3/imsatvat_0.25 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=0.25}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_4/imsatvat_0.25 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=0.25}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_5/imsatvat_0.25 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=0.25}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_1/imsatvat_1.0 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=1.0}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_2/imsatvat_1.0 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=1.0}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_3/imsatvat_1.0 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=1.0}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_4/imsatvat_1.0 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=1.0}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_5/imsatvat_1.0 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=1.0}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_1/imsatvat_2.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=2.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_2/imsatvat_2.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=2.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_3/imsatvat_2.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=2.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_4/imsatvat_2.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=2.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_5/imsatvat_2.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=2.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_1/imsatvat_5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_2/imsatvat_5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_3/imsatvat_5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_4/imsatvat_5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_5/imsatvat_5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_1/imsatvat_7.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=7.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_2/imsatvat_7.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=7.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_3/imsatvat_7.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=7.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_4/imsatvat_7.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=7.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_5/imsatvat_7.5 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=7.5}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_1/imsatvat_10 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=10}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_2/imsatvat_10 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=10}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_3/imsatvat_10 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=10}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_4/imsatvat_10 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=10}'" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=mnist_5/imsatvat_10 Trainer.max_epoch=100 Trainer.VAT_params='{name:kl, eps=10}'" \
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${cmd}"
#${cmd}
done
