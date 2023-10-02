import numpy as np
import h5py
from keras.layers import Input, Dense
from keras.models import Model
import os
import sys

from orcanet.core import Organizer
from orcanet.core import Configuration

def use_orcanet():

    inputfile   = str(sys.argv[1])
    outputfile  = str(sys.argv[2])
    temp_folder = str(sys.argv[3])
    config_file = str(sys.argv[4])
    
    organizer = Organizer(temp_folder, config_file=config_file)
    
    organizer.inference_on_file(inputfile, outputfile)

use_orcanet()
