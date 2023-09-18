# ####################################################################################################################
# perform_basset_augmentation_analysis.py
#
# Train model on the Basset data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import models_drosophila as models

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
model_types = ['deepstarr', 'motif_deepstarr', 'explainn']
num_replicates = 3
sample_fraction = 1.0

file_folder = "../input/basset/"
homolog_folder = "../input/basset/basset_homologs/"
output_folder = "../output/basset_augmentation/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, model_type, replicate):
    models.train_basset(use_homologs, sample_fraction, replicate, file_folder,
                        homolog_folder, output_folder)

    models.fine_tune_basset(use_homologs, sample_fraction, replicate, file_folder,
                            homolog_folder, output_folder)


for model_type in model_types:
    for replicate in range(1, num_replicates + 1):
        train_model(True, model_type, replicate)
        train_model(False, model_type, replicate)
