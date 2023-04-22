# ####################################################################################################################
# perform_drosophila_augmentation_analysis.py
#
# Train model using the STARR-seq data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))

file_folder = "../process_data/drosophila/output/"
homolog_folder = "../process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila/"
tasks = ['Dev', 'Hk']
sequence_size = 249

# ====================================================================================================================
# Main code
# ====================================================================================================================
if model_type == "deepstarr":
	models.train_deepstarr(use_homologs, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type)
elif model_type == "explainn":
	models.train_explainn(use_homologs, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type)
elif model_type == "motif_deepstarr":
	models.train_motif_deepstarr(use_homologs, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type)