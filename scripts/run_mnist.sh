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


account=def-mpederso
time=1
transforms=strong
main_dir="07_15_benchmark/${transforms}"
vat_eps=10
max_epoch=1000
seed=1


declare -a StringArray=(
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeo Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicmixup Trainer.save_dir=${main_dir}/mnist_${seed}/iicmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicvat Trainer.save_dir=${main_dir}/mnist_${seed}/iicvat_kl Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Trainer.VAT_params.name=kl" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovat Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeovat_kl Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Trainer.VAT_params.name=kl" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeomixup Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovatmixup Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeovatmixup_kl Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Trainer.VAT_params.name=kl" \

"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicvat Trainer.save_dir=${main_dir}/mnist_${seed}/iicvat_mi Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Trainer.VAT_params.name=mi" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovat Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeovat_mi Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Trainer.VAT_params.name=mi" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovatmixup Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeovatmixup_mi Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Trainer.VAT_params.name=mi" \

"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovatreg Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeovatreg Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeomixupreg Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeomixupreg Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovatmixupreg Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeovatmixupreg Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=iicgeovatvatreg Trainer.save_dir=${main_dir}/mnist_${seed}/iicgeovatvatreg_kl Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \

"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=imsat Trainer.save_dir=${main_dir}/mnist_${seed}/imsat Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=imsatvat Trainer.save_dir=${main_dir}/mnist_${seed}/imsatvat Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=imsatgeo Trainer.save_dir=${main_dir}/mnist_${seed}/imsatgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=imsatmixup Trainer.save_dir=${main_dir}/mnist_${seed}/imsatmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatmixup Trainer.save_dir=${main_dir}/mnist_${seed}/imsatvatmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeo Trainer.save_dir=${main_dir}/mnist_${seed}/imsatvatgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=imsatgeomixup Trainer.save_dir=${main_dir}/mnist_${seed}/imsatgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
"python -O  main.py Config=config/config_MNIST.yaml Trainer.name=imsatvatgeomixup Trainer.save_dir=${main_dir}/mnist_${seed}/imsatvatgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms}" \
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${account}" "${cmd}"
# ${cmd}
done
