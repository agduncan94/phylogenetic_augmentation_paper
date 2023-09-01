# ####################################################################################################################
# perform_drosophila_homolog_rate_analysis.py
#
# Train model on the Drosophila S2 STARR-seq data with varying homolog rates
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
homolog_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sample_fraction = 1.0

file_folder = "../analysis/process_data/drosophila/output/"
homolog_folder = "../analysis/process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila_homolog_rate/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, model_type, replicate, homolog_rate):
    if model_type == "deepstarr":
        models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                               homolog_folder, output_folder, homolog_rate=homolog_rate)
    elif model_type == "explainn":
        models.train_explainn(use_homologs, sample_fraction, replicate, file_folder,
                              homolog_folder, output_folder, homolog_rate=homolog_rate)
    elif model_type == "motif_deepstarr":
        models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                                     homolog_folder, output_folder, homolog_rate=homolog_rate)


# Train models with increasing homolog rates
for model_type in model_types:
    for homolog_rate in homolog_rates:
        for replicate in range(1, num_replicates + 1):
            train_model(True, model_type, replicate, homolog_rate)

# Train model with no augmentation
for model_type in model_types:
    for replicate in range(1, num_replicates + 1):
        train_model(False, model_type, replicate, 1.0)
