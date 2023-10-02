#!/bin/bash

singularity_image="/sps/km3net/users/fcarenin/GNN/orcanet_v1.0.4.sif"

singularity exec --nv --bind /pbs:/pbs --bind /sps:/sps ${singularity_image} /sps/km3net/users/fcarenin/GNN/scripts/inference/run_inference.sh
