# ####################################################################################################################
# perform_drosophila_sampling_analysis.py
#
# Train model using the STARR-seq data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models_drosophila as models

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))
sample_fraction = float(sys.argv[4])

file_folder = "../process_data/drosophila/output/"
homolog_folder = "../process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila_sampling/"

# ====================================================================================================================
# Main code
# ====================================================================================================================
if model_type == "deepstarr":
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder)
elif model_type == "explainn":
    models.train_explainn(use_homologs, sample_fraction, replicate, file_folder,
                          homolog_folder, output_folder)
elif model_type == "motif_deepstarr":
    models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                                 homolog_folder, output_folder)
