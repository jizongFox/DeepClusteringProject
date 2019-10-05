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
main_dir="wacv_resivion_best_model/${transforms}"
max_epoch=1000
seed=1
load_checkpoint="#"
head_A=2
head_B=1
subhead_num=5

declare -a StringArray=(

# IIC+Reg series
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatreg_kl \
Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms} Arch.num_sub_heads=${subhead_num} Trainer.head_control_params.A=${head_A} Trainer.head_control_params.B=${head_B} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatreg_kl" \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeomixupreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeomixupreg \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Arch.num_sub_heads=${subhead_num} Trainer.head_control_params.A=${head_A} Trainer.head_control_params.B=${head_B} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeomixupreg" \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeocutoutreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeocutoutreg  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Arch.num_sub_heads=${subhead_num} Trainer.head_control_params.A=${head_A} Trainer.head_control_params.B=${head_B} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeocutoutreg " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeogaussianreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeogaussianreg  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Arch.num_sub_heads=${subhead_num} Trainer.head_control_params.A=${head_A} Trainer.head_control_params.B=${head_B} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeogaussianreg " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatcutoutreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatcutoutreg  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Arch.num_sub_heads=${subhead_num} Trainer.head_control_params.A=${head_A} Trainer.head_control_params.B=${head_B} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatcutoutreg " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatgaussianreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatgaussianreg  \
Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} Arch.num_sub_heads=${subhead_num} Trainer.head_control_params.A=${head_A} Trainer.head_control_params.B=${head_B} \
${load_checkpoint}Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatgaussianreg " \

)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${account}" "${cmd}"
# ${cmd}
done


