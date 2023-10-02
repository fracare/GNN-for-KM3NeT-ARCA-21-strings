#!/bin/bash 

export DIR=/sps/km3net/users/fcarenin/GNN/

sbatch -L sps --output=/sps/km3net/users/fcarenin/GNN/logs/log_concatenate.log --time 1-00:00 -n 1 --mem=10G $DIR/scripts/orcasong/concatenate.sh 
