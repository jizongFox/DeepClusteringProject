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
main_dir="07_15_benchmark/${transforms}"
max_epoch=2
seed=1


declare -a StringArray=(
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeo Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeo" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicmixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicmixup" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicvat Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicvat_kl Trainer.VAT_params.name=kl Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicvat_kl " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovat Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovat_kl Trainer.VAT_params.name=kl Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovat_kl " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeomixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeomixup" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatmixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatmixup_kl Trainer.max_epoch=${max_epoch} Trainer.VAT_params.name=kl Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatmixup_kl " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicvat Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicvat_mi Trainer.VAT_params.name=mi Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicvat_mi " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovat Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovat_mi Trainer.VAT_params.name=mi Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovat_mi " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatmixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatmixup_mi Trainer.max_epoch=${max_epoch} Trainer.VAT_params.name=mi Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatmixup_mi " \

"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatreg_kl Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatreg_kl" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatreg_mi Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=mi DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatreg_mi " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeomixupreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeomixupreg Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeomixupreg" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatmixupreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatmixupreg_kl Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatmixupreg_kl " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatmixupreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatmixupreg_mi Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=mi DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatmixupreg_mi " \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=iicgeovatvatreg Trainer.save_dir=${main_dir}/${DATASET}_${seed}/iicgeovatvatreg_kl Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/iicgeovatvatreg_kl " \


"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsat Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsat Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsat" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvat Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvat Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvat" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatgeo Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatgeo" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatmixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatmixup" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvatmixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvatmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvatmixup" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvatgeo Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvatgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvatgeo" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatgeomixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatgeomixup" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvatgeomixup Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvatgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvatgeomixup" \
"python -O main.py Config=config/config_${DATASET}.yaml Trainer.name=imsatvatiicgeo Trainer.save_dir=${main_dir}/${DATASET}_${seed}/imsatvatiicgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/${DATASET}_${seed}/imsatvatiicgeo" \

)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${account}" "${cmd}"
# ${cmd}
done
