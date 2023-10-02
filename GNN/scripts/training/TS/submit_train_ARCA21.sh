#!/bin/bash

export unique_name="with16neighb"
export no_of_epochs=40
export det="ARCA21"
export to_determine="ts_onlynu_corrected"

export output_dir="/sps/km3net/users/fcarenin/GNN/${det}/${to_determine}/${unique_name}/"
mkdir -p $output_dir

job_name="${det}_train_${to_determine}_${unique_name}"

script="/sps/km3net/users/fcarenin/GNN/scripts/training/batch_train_time_window.sh"

#copy the configs and stuff not to get mixed up
export scripts_dir="${output_dir}scripts"
mkdir -p ${scripts_dir}

export list_file=listARCA21.toml
export config_file=configARCA21.toml
export model_file=modelARCA21.toml

cp /sps/km3net/users/fcarenin/GNN/scripts/training/TS/$list_file ${scripts_dir}/
cp /sps/km3net/users/fcarenin/GNN/scripts/training/TS/$config_file ${scripts_dir}/
cp /sps/km3net/users/fcarenin/GNN/scripts/training/TS/$model_file ${scripts_dir}/


sbatch  -p gpu --gres=gpu:v100:1 --time 7-00:00 -n 1 --mem 10G  --job-name=${job_name} --output=/sps/km3net/users/fcarenin/GNN/logs/log_arca6_16neighb.log ${script}


