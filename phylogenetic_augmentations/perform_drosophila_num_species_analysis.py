# ####################################################################################################################
# perform_drosophila_num_species_analysis.py
#
# Train model on the Drosophila S2 STARR-seq data with varying numbers of species
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models_drosophila as models
import pandas as pd

# ====================================================================================================================
# Variables
# ====================================================================================================================
model_types = ['deepstarr']
num_replicates = 3
sample_fraction = 1.0
num_species = 20
species_file = "../analysis/process_data/drosophila/output/ordered_drosophila_species.txt"

file_folder = "../analysis/process_data/drosophila/output/"
homolog_folder = "../analysis/process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila_num_species/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, model_type, replicate, species_list):
    if model_type == "deepstarr":
        models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                               homolog_folder, output_folder, species=species_list)
    elif model_type == "explainn":
        models.train_explainn(use_homologs, sample_fraction, replicate, file_folder,
                              homolog_folder, output_folder, species=species_list)
    elif model_type == "motif_deepstarr":
        models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                                     homolog_folder, output_folder, species=species_list)


for model_type in model_types:
    for count in range(1, num_species + 1):
        # Read species file
        species_df = pd.read_csv(species_file, sep='\t')
        species_list = species_df['species'].to_list()

        # species_list.reverse()

        species_list = species_list[0:count]
        for replicate in range(1, num_replicates + 1):
            train_model(True, model_type, replicate, species_list)
            train_model(False, model_type, replicate, species_list)
