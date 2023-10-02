#!/usr/bin/bash

while read p

do
    echo $p
    python /pbs/throng/km3net/software/python/3.7.5/lib/python3.7/site-packages/km3pipe/utils/h5extractf.py  /sps/km3net/users/fcarenin/GNN/files_valid_more_tracks/root/$p

done < list_valid.txt
