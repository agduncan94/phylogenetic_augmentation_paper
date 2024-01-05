# ####################################################################################################################
# perform_utr_augmentation_analysis.py
#
# Train model on the 3'UTR data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import models_utr as models

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
model_types = ['deepstarr']

num_replicates = 1
sample_fraction = 1.0

file_folder = "../input/"
homolog_folder = "../input/yeast_homologs/"
output_folder = "../output/yeast_augmentation_deepstarr_update_metrics/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, model_type, replicate):
    if model_type == "deepstarr":
        models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                               homolog_folder, output_folder)


for model_type in model_types:
    for replicate in range(1, num_replicates + 1):
        train_model(False, model_type, replicate)
        train_model(True, model_type, replicate)
