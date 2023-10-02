#!/usr/bin/python -i
from orcasong.core import FileGraph
from orcasong.extractors import get_muon_mc_info_extr,get_neutrino_mc_info_extr,get_real_data_info_extr,get_random_noise_mc_info_extr
import numpy as np
import sys
import os

filelist = str(sys.argv[1])
#label = str(sys.argv[2])
path = str(sys.argv[2])

inputfile = np.loadtxt(filelist, dtype=str)

for i in range(len(inputfile)):

    print("Processing file:",inputfile[i])
    if('mup' in inputfile[i]):
        fg = FileGraph(max_n_hits=5000, extractor=get_muon_mc_info_extr(path + str(inputfile[i])),keep_event_info = True)
    elif('gsg' in inputfile[i]):
        fg = FileGraph(max_n_hits=5000, extractor=get_neutrino_mc_info_extr(path + str(inputfile[i])),keep_event_info = True)

#    if(label == "muon"):
#        fg = FileGraph(max_n_hits=5000, extractor=get_muon_mc_info_extr(path + str(inputfile[i])),keep_event_info = True)
#    elif(label == "nu"):
#        fg = FileGraph(max_n_hits=5000, extractor=get_neutrino_mc_info_extr(path + str(inputfile[i])),keep_event_info = True)
#    elif(label == "real"):
#        fg = FileGraph(max_n_hits=5000, extractor=get_real_data_info_extr(path + str(inputfile[i])),keep_event_info = True)
#    elif(label == "noise"):
#        fg = FileGraph(max_n_hits=5000, extractor=get_random_noise_mc_info_extr(path + str(inputfile[i])),keep_event_info = True)

    else:
        print("Insert the correct data label among: muon, nu, real and noise.")
        exit()

    fg.run(path + inputfile[i])
