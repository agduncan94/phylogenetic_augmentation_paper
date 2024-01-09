#!/bin/bash

# ####################################################################################################################
# download_input_data.py
#
# Downloads the necessary input files for running the analysis from the paper
# ####################################################################################################################

# Download data from Zenodo (https://zenodo.org/record/8356747)

# Drosophila data
wget https://zenodo.org/record/8356747/files/Drosophila_Sequences_Train.txt
wget https://zenodo.org/record/8356747/files/Drosophila_Sequences_Val.txt
wget https://zenodo.org/record/8356747/files/Drosophila_Sequences_Test.txt
wget https://zenodo.org/record/8356747/files/ordered_drosophila_species.txt
wget https://zenodo.org/record/8356747/files/all_drosphila_species_distances_asc.txt
wget https://zenodo.org/record/8356747/files/drosophila_homologs.zip
unzip drosophila_homologs.zip

# Basset data
wget https://zenodo.org/record/8356747/files/Basset_Sequences_Train.txt
wget https://zenodo.org/record/8356747/files/Basset_Sequences_Val.txt
wget https://zenodo.org/record/8356747/files/Basset_Sequences_Test.txt
wget https://zenodo.org/record/8356747/files/augmentation_data_homologs.hdf5

# 3' UTR data
wget https://zenodo.org/record/8356747/files/Yeast_Sequences_Train.txt
wget https://zenodo.org/record/8356747/files/Yeast_Sequences_Test.txt
wget https://zenodo.org/record/8356747/files/yeast_homologs.zip
unzip yeast_homologs.zip

