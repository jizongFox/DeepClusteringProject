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

account=def-mpederso  #rrg-mpederso, def-mpederso, and def-chdesa
time=12
DATASET=MNIST
transforms=strong
main_dir="wacv_revision/${transforms}"
max_epoch=200
seed=1


declare -a StringArray=(

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgaussian Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgaussian/std_005 Trainer.Gaussian_params.gaussian_std=0.05 Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgaussian/std_005 " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgaussian Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgaussian/std_010 Trainer.Gaussian_params.gaussian_std=0.10 Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgaussian/std_010 " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatgaussian Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatgaussian/std_005 Trainer.Gaussian_params.gaussian_std=0.05 Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatgaussian/std_005 " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatgaussian Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatgaussian/std_010 Trainer.Gaussian_params.gaussian_std=0.10 Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatgaussian/std_010 " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iiccutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iiccutout  Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iiccutout " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatcutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatcutout  Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatcutout " \

)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${account}" "${cmd}"
# ${cmd}
done
