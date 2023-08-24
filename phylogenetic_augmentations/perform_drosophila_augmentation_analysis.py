# ####################################################################################################################
# perform_drosophila_augmentation_analysis.py
#
# Train model on the Drosophila S2 STARR-seq data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models_drosophila as models

# ====================================================================================================================
# Variables
# ====================================================================================================================
model_types = ['deepstarr', 'motif_deepstarr', 'explainn']
num_replicates = 3

file_folder = "../analysis/process_data/drosophila/output/"
homolog_folder = "../analysis/process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila/"
sample_fraction = 1.0

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, model_type, replicate):
    if model_type == "deepstarr":
        models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                               homolog_folder, output_folder)
    elif model_type == "explainn":
        models.train_explainn(use_homologs, sample_fraction, replicate, file_folder,
                              homolog_folder, output_folder)
    elif model_type == "motif_deepstarr":
        models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                                     homolog_folder, output_folder)


for model_type in model_types:
    for replicate in range(1, num_replicates + 1):
        train_model(True, model_type, replicate)
        train_model(False, model_type, replicate)
