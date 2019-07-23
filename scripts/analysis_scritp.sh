#!/usr/bin/env bash
main_folder=/home/jizong/Workspace/DeepClusteringProject/runs/07_15_benchmark/strong
folder_lists=$(find ${main_folder} -mindepth 2 -maxdepth 2 -type d -print)
cd ..
for folder in ${folder_lists}
do
echo $folder
python analyze_main.py checkpoint_path=${folder}
done

