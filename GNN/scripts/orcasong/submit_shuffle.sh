#!/bin/bash 

export DIR=/sps/km3net/users/fcarenin/GNN/

#$DIR/OrcaSong/shuffle.sh

sbatch -L sps --output=/sps/km3net/users/fcarenin/GNN/logs/log_shuf.log --time 1-00:00 -n 1 --mem=10G $DIR/scripts/orcasong/shuffle.sh 
