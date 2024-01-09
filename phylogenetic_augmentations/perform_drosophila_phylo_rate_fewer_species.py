# ####################################################################################################################
# perform_drosophila_homolog_rate_analysis.py
#
# Train model on the Drosophila S2 STARR-seq data with varying homolog rates
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import models_drosophila as models
import pandas as pd

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
model_types = ['deepstarr']
num_replicates = 3
homolog_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

sample_fraction = 1.0

file_folder = "../input/"
homolog_folder = "../input/drosophila_homologs/"
species_file = "../input/ordered_drosophila_species_only.txt"
output_folder = "../output/drosophila_phylo_aug_rate_reduced_species/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, model_type, replicate, homolog_rate, species_list):
    if model_type == "deepstarr":
        models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                               homolog_folder, output_folder, phylo_aug_rate=homolog_rate, species=species_list)

# Train models with increasing homolog rates
species_df = pd.read_csv(species_file, sep='\t')
species_list = species_df['species'].to_list()
species_list = species_list[0:10]

for model_type in model_types:
    for homolog_rate in homolog_rates:
        for replicate in range(1, num_replicates + 1):
            train_model(True, model_type, replicate, homolog_rate, species_list)

# Train model with no augmentation
for model_type in model_types:
    for replicate in range(1, num_replicates + 1):
        train_model(False, model_type, replicate, 1.0, None)
