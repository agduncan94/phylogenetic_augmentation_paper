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
num_species = [1, 5, 10, 20, 136]

model_types = ['deepstarr']

file_folder = "../input/"
homolog_folder = "../input/drosophila_homologs/"
species_file = "../input/ordered_drosophila_species_only.txt"
output_folder = "../output/drosophila_num_species_rev/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, replicate, species_list):
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, species=species_list)


for model_type in model_types:
    for count in num_species:
        # Get a list of species
        species_df = pd.read_csv(species_file, sep='\t')
        species_list = species_df['species'].to_list()
        species_list.reverse()
        species_list = species_list[0:count]

        # Train multiple replicates of model
        for replicate in range(1, num_replicates + 1):
            train_model(True, replicate, species_list)
