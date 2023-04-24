# ####################################################################################################################
# perform_chef_augmentation_analysis.py
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

file_folder = "../process_data/chef/output/"
homolog_folder = "../process_data/chef/output/orthologs/"
output_folder = "./output_chef/"
tasks = ['h3k27ac_log2_enrichment', 'tf_sum']
sequence_size = 700
sample_fraction = 1.0

# ====================================================================================================================
# Main code
# ====================================================================================================================
if model_type == "deepstarr":
	models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type)
elif model_type == "explainn":
	models.train_explainn(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type)
elif model_type == "motif_deepstarr":
	models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type)