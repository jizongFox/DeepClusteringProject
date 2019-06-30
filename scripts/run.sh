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
time=24

declare -a StringArray=(
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=update_imsat/mnist_1/iicgeo Trainer.max_epoch=500" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=update_imsat/mnist_1/imsat Trainer.max_epoch=500" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_0.25 Trainer.max_epoch=500 Trainer.VAT_params={eps:0.25}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_01 Trainer.max_epoch=500 Trainer.VAT_params={eps:1}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_2.5 Trainer.max_epoch=500 Trainer.VAT_params={eps:2.5}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_05 Trainer.max_epoch=500 Trainer.VAT_params={eps:5}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_10 Trainer.max_epoch=500 Trainer.VAT_params={eps:10}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeo Trainer.save_dir=update_imsat/mnist_1/imsatvatgeo_0.25 Trainer.max_epoch=500 Trainer.VAT_params={eps:0.25}" \ 
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeo Trainer.save_dir=update_imsat/mnist_1/imsatvatgeo_01 Trainer.max_epoch=500 Trainer.VAT_params={eps:1}" \ 
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeo Trainer.save_dir=update_imsat/mnist_1/imsatvatgeo_2.5 Trainer.max_epoch=500 Trainer.VAT_params={eps:2.5}" \ 
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeo Trainer.save_dir=update_imsat/mnist_1/imsatvatgeo_05 Trainer.max_epoch=500 Trainer.VAT_params={eps:5}" \ 
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeo Trainer.save_dir=update_imsat/mnist_1/imsatvatgeo_10 Trainer.max_epoch=500 Trainer.VAT_params={eps:10}" \ 
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${cmd}"
#${cmd}
done
