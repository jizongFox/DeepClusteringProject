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
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=update_imsat/mnist_1/iicgeo Trainer.max_epoch=1000" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=update_imsat/mnist_1/imsat Trainer.max_epoch=1000" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_0.25_MI_4 Trainer.max_epoch=1000 Trainer.VAT_params={eps:0.25} Trainer.MI_params={mu:4.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_01_MI_4 Trainer.max_epoch=1000 Trainer.VAT_params={eps:1} Trainer.MI_params={mu:4.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_2.5_MI_4 Trainer.max_epoch=1000 Trainer.VAT_params={eps:2.5} Trainer.MI_params={mu:4.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_05_MI_4 Trainer.max_epoch=1000 Trainer.VAT_params={eps:5} Trainer.MI_params={mu:4.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_10_MI_4 Trainer.max_epoch=1000 Trainer.VAT_params={eps:10} Trainer.MI_params={mu:4.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_0.25_MI_8 Trainer.max_epoch=1000 Trainer.VAT_params={eps:0.25} Trainer.MI_params={mu:8.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_01_MI_8 Trainer.max_epoch=1000 Trainer.VAT_params={eps:1} Trainer.MI_params={mu:8.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_2.5_MI_8 Trainer.max_epoch=1000 Trainer.VAT_params={eps:2.5} Trainer.MI_params={mu:8.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_05_MI_8 Trainer.max_epoch=1000 Trainer.VAT_params={eps:5} Trainer.MI_params={mu:8.0}" \
"python -O main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=update_imsat/mnist_1/imsatvat_10_MI_8 Trainer.max_epoch=1000 Trainer.VAT_params={eps:10} Trainer.MI_params={mu:8.0}" \
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
#wrapper "${time}" "${cmd}"
${cmd}
done
