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
"python  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=test/mnist_1/iicgeo Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=iicmixup Trainer.save_dir=test/mnist_1/iicmixup Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=iicvat Trainer.save_dir=test/mnist_1/iicvat Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovat Trainer.save_dir=test/mnist_1/iicgeovat Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovatmixup Trainer.save_dir=test/mnist_1/iicgeovatmixup Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=test/mnist_1/imsat Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=test/mnist_1/imsatvat Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=imsatmixup Trainer.save_dir=test/mnist_1/imsatmixup Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatmixup Trainer.save_dir=test/mnist_1/imsatvatmixup Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeo Trainer.save_dir=test/mnist_1/imsatvatgeo Trainer.max_epoch=2 Seed=1" \
"python  main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeomixup Trainer.save_dir=test/mnist_1/imsatvatgeomixup Trainer.max_epoch=2 Seed=1" \
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
#wrapper "${time}" "${cmd}"
${cmd}
done
