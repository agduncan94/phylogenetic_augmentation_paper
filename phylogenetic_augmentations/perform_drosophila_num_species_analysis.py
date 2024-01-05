# ####################################################################################################################
# perform_drosophila_num_species_analysis.py
#
# Train DeepSTARR model on the Drosophila S2 STARR-seq data with varying numbers of species
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import models_drosophila as models
import pandas as pd

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
num_replicates = 3
sample_fraction = 1
num_species = 20
model_types = ['deepstarr']
reverse = False

file_folder = "../input/"
homolog_folder = "../input/drosophila_homologs/"
species_file = "../input/ordered_drosophila_species_only.txt"
output_folder = "../output/drosophila_num_species/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, replicate, species_list):
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, species=species_list)


for model_type in model_types:
    for count in range(1, num_species + 1):
        # Read species file
        species_df = pd.read_csv(species_file, sep='\t')
        species_list = species_df['species'].to_list()
        if reverse:
            species_list.reverse() # Reverse order

        species_list = species_list[0:count]
        for replicate in range(1, num_replicates + 1):
            train_model(True, replicate, species_list)
            #train_model(False, replicate, species_list)
