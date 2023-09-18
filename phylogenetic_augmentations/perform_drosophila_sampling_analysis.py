# ####################################################################################################################
# perform_drosophila_sampling_analysis.py
#
# Train model on the Drosophila S2 STARR-seq data with varying fractions
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import models_drosophila as models

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
model_types = ['deepstarr']
num_replicates = 3
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

file_folder = "../input/drosophila/"
homolog_folder = "../input/drosophila/drosophila_homologs/"
output_folder = "../output/drosophila_sampling/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, model_type, replicate, fraction):
    if model_type == "deepstarr":
        models.train_deepstarr(use_homologs, fraction, replicate, file_folder,
                               homolog_folder, output_folder)
    elif model_type == "explainn":
        models.train_explainn(use_homologs, fraction, replicate, file_folder,
                              homolog_folder, output_folder)
    elif model_type == "motif_deepstarr":
        models.train_motif_deepstarr(use_homologs, fraction, replicate, file_folder,
                                     homolog_folder, output_folder)


for model_type in model_types:
    for fraction in fractions:
        for replicate in range(1, num_replicates + 1):
            train_model(True, model_type, replicate, fraction)
            train_model(False, model_type, replicate, fraction)
