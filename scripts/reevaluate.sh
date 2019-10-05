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
time=24
DATASET=MNIST
transforms=strong
main_dir="wacv_reevaluation_1005/${transforms}"
max_epoch=1000
seed=1
load_checkpoint="#"


declare -a StringArray=(

# IIC series
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeo Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeo \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeo" \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeomixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeomixup \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeomixup" \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeogaussian Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeogaussian  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeogaussian/ " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeocutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeocutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeocutout " \

# IIC+Reg series
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatreg_kl \
Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatreg_kl" \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeomixupreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeomixupreg \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeomixupreg" \

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

# imsat series
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvat Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvat \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvat" \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvatmixupcutout Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvatmixupcutout  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvatmixupcutout " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvatiicgeo Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvatiicgeo \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvatiicgeo"
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${account}" "${cmd}"
# ${cmd}
done
