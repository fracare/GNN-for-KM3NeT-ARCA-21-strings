#!/usr/bin/python -i
from orcasong.core import FileGraph
from orcasong.extractors import get_muon_mc_info_extr,get_neutrino_mc_info_extr,get_real_data_info_extr,get_random_noise_mc_info_extr
import numpy as np
import sys
import os

inputfile = str(sys.argv[1])
label = str(sys.argv[2])
#detectorfile = str(sys.argv[3])
#outputfile = str(sys.argv[4])

#inputfile = np.loadtxt(filename, dtype=str)
#path = '/training/ARCA21/'

if(label == "muon"):
    fg = FileGraph(max_n_hits=5000, extractor=get_muon_mc_info_extr(inputfile),keep_event_info = True)
elif(label == "nu"):
    fg = FileGraph(max_n_hits=5000, extractor=get_neutrino_mc_info_extr(inputfile),keep_event_info = True)
elif(label == "real"):
#    fg = FileGraph(max_n_hits=5000, extractor=get_real_data_info_extr(inputfile),det_file=detectorfile,keep_event_info = True)
    fg = FileGraph(max_n_hits=5000, extractor=get_real_data_info_extr(inputfile),keep_event_info = True)
elif(label == "noise"):
    fg = FileGraph(max_n_hits=5000, extractor=get_random_noise_mc_info_extr(inputfile),keep_event_info = True)
else:
    print("Insert the correct data label among: muon, nu, real and noise.")
    exit()

if(len(sys.argv) == 4):
    outputfile = str(sys.argv[3])
    fg.run(inputfile,outputfile)
else:
    fg.run(inputfile)
