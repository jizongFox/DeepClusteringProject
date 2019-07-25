#!/usr/bin/env bash
main_folder=/home/jizong/Workspace/DeepClusteringProject/runs/mnist/iicgeo
folder_lists=$(find ${main_folder} -mindepth 0 -maxdepth 0 -type d -print)
cd ..
for folder in ${folder_lists}
do
echo $folder
python analyze_main.py checkpoint_path=${folder}
done

