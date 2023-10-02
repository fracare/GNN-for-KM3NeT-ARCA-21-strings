#!/bin/bash

script="/sps/km3net/users/fcarenin/GNN/scripts/inference/batch_inference.sh"

sbatch  -p gpu --gres=gpu:v100:1 --time 7-00:00 -n 1 --mem 10G  --job-name='ARCA21' --output=/sps/km3net/users/fcarenin/GNN/logs/log_arca21_test.log ${script}

