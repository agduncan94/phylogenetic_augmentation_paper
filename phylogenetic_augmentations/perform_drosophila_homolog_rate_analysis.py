# ####################################################################################################################
# perform_drosophila_homolog_rate_analysis.py
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
homolog_rate = float(sys.argv[4])

file_folder = "../process_data/drosophila/output/"
homolog_folder = "../process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila_homolog_rate/"
sample_fraction = 1.0

# ====================================================================================================================
# Main code
# ====================================================================================================================
if model_type == "deepstarr":
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, homolog_rate=homolog_rate)
elif model_type == "explainn":
    models.train_explainn(use_homologs, sample_fraction, replicate, file_folder,
                          homolog_folder, output_folder, homolog_rate=homolog_rate)
elif model_type == "motif_deepstarr":
    models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                                 homolog_folder, output_folder, homolog_rate=homolog_rate)
elif model_type == "motif_linear":
    models.train_motif_linear(use_homologs, sample_fraction, replicate, file_folder,
                              homolog_folder, output_folder, homolog_rate=homolog_rate)
elif model_type == "motif_linear_relu":
    models.train_motif_linear_relu(use_homologs, sample_fraction, replicate, file_folder,
                                   homolog_folder, output_folder, homolog_rate=homolog_rate)
