# GNN-for-KM3NeT-ARCA-21-strings
This repository contains all the scripts needed to run a track and shower classification with GNN for the detector KM3NeT/ARCA. In order to do this, the installation of some packages is needed, namely Orcasong (https://ml.pages.km3net.de/OrcaSong/index.html) and Orcanet (https://ml.pages.km3net.de/OrcaNet/index.html). 

# Notes:

1. As singularity image for the analysis I used the orcanet_v1.0.4.sif image, details available at https://ml.pages.km3net.de/OrcaNet/index.html#containerization.
2. The overall size of files (.root, .h5 and root_dl.h5) used for the analysis exceeded 100GB so I did not uploaded them on github. Abundances are specified in GNN_KM3NeT_ARCA_carenini.pdf.
3. ARCA21_TS.h5 is the final output file, after inference step, containing information about events scores. The one reported in this git repository is referred to the model of this analysis. 
