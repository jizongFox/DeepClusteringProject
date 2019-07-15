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
time=12
transforms=strong
main_dir="07_15_benchmark/${transforms}"
vat_eps=10
iic_vat_name=kl
max_epoch=1000
seed=1
declare -a StringArray=(
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeo Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeo" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicmixup Trainer.save_dir=${main_dir}/cifar_${seed}/iicmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicmixup" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicvat Trainer.save_dir=${main_dir}/cifar_${seed}/iicvat_kl Trainer.VAT_params.name=kl Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicvat_kl " \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovat Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovat_kl Trainer.VAT_params.name=kl Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovat_kl " \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeomixup Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeomixup" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovatmixup Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovatmixup_kl Trainer.max_epoch=${max_epoch} Trainer.VAT_params.name=kl Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovatmixup_kl " \

"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicvat Trainer.save_dir=${main_dir}/cifar_${seed}/iicvat_mi Trainer.VAT_params.name=mi Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicvat_mi " \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovat Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovat_mi Trainer.VAT_params.name=mi Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovat_mi " \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovatmixup Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovatmixup_mi Trainer.max_epoch=${max_epoch} Trainer.VAT_params.name=mi Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovatmixup_mi " \

"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovatreg Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovatreg_kl Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovatreg_kl" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovatreg Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovatreg_mi Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=mi DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovatreg_mi " \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeomixupreg Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeomixupreg Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeomixupreg" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovatmixupreg Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovatmixupreg_kl Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovatmixupreg_kl " \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovatmixupreg Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovatmixupreg_mi Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=mi DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovatmixupreg_mi " \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=iicgeovatvatreg Trainer.save_dir=${main_dir}/cifar_${seed}/iicgeovatvatreg_kl Trainer.max_epoch=${max_epoch} Seed=${seed} Trainer.VAT_params.name=kl DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/iicgeovatvatreg_kl " \


"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsat Trainer.save_dir=${main_dir}/cifar_${seed}/imsat Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/imsat" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvat Trainer.save_dir=${main_dir}/cifar_${seed}/imsatvat Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/imsatvat" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatgeo Trainer.save_dir=${main_dir}/cifar_${seed}/imsatgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/imsatgeo" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatmixup Trainer.save_dir=${main_dir}/cifar_${seed}/imsatmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/imsatmixup" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvatmixup Trainer.save_dir=${main_dir}/cifar_${seed}/imsatvatmixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/imsatvatmixup" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvatgeo Trainer.save_dir=${main_dir}/cifar_${seed}/imsatvatgeo Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/imsatvatgeo" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatgeomixup Trainer.save_dir=${main_dir}/cifar_${seed}/imsatgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/imsatgeomixup" \
"python -O main.py Config=config/config_CIFAR.yaml Trainer.name=imsatvatgeomixup Trainer.save_dir=${main_dir}/cifar_${seed}/imsatvatgeomixup Trainer.max_epoch=${max_epoch} Seed=${seed} DataLoader.transforms=${transforms} #Trainer.checkpoint_path=runs/${main_dir}/cifar_${seed}/imsatvatgeomixup" \
)
#
for cmd in "${StringArray[@]}"
do
echo ${cmd}
wrapper "${time}" "${cmd}"
# ${cmd}
done
