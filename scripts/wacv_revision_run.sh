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
max_epoch=1
seed=1
load_checkpoint=""


declare -a StringArray=(

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatgaussian Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatgaussian  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatgaussian/ " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatcutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatcutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatcutout " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatcutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatcutoutgaussian  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatcutoutgaussian "

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvatcutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvatcutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvatcutout "


"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatmixupcutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatmixupcutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatmixupcutout "


"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvatmixupcutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvatmixupcutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvatmixupcutout "

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatgeovatcutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatgeovatcutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatgeovatcutout "





"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iiccutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iiccutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iiccutout " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgaussian Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgaussian  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgaussian/ " \


"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeocutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeocutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeocutout " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeogaussian Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeogaussian  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeogaussian/ " \




"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeocutoutreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeocutoutreg  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeocutoutreg " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeogaussianreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeogaussianreg  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeogaussianreg " \


"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatcutoutreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatcutoutreg  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatcutoutreg " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatgaussianreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatgaussianreg  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatgaussianreg " \

)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
#wrapper "${time}" "${account}" "${cmd}"
 ${cmd}
done
