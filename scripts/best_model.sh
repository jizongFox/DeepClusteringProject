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
main_dir="best_model/${transforms}"
max_epoch=2000
seed=1


declare -a StringArray=(
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatreg_kl \
Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms}  Arch.num_sub_heads=5 Trainer.head_control_params.A=1 Trainer.head_control_params.B=2 \
#Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatreg_kl"
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${account}" "${cmd}"
# ${cmd}
done


