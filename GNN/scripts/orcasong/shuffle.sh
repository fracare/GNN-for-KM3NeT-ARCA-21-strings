#!/bin/bash

echo "START"

~/.local/bin/orcasong h5shuffle2 --output_file $DIR/scripts/orcasong/muon_and_nu_shuffled.h5 $DIR/scripts/orcasong/muon_and_nu.h5 --max_ram 500000000

echo "END"
