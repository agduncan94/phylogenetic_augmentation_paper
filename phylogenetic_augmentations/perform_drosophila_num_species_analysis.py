# ####################################################################################################################
# perform_drosophila_num_species_analysis.py
#
# Train model using the STARR-seq data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models_drosophila as models
import pandas as pd

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))
num_species = int(sys.argv[4])

file_folder = "../process_data/drosophila/output/"
homolog_folder = "../process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila_num_species_rev/"
sample_fraction = 1.0
species_file = "../process_data/drosophila/output/ordered_drosophila_species.txt"

# ====================================================================================================================
# Main code
# ====================================================================================================================

# Read species file
species_df = pd.read_csv(species_file, sep='\t')
species_list = species_df['species'].to_list()

species_list.reverse()

species_list = species_list[0:num_species]


if model_type == "deepstarr":
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, species=species_list)
