# ####################################################################################################################
# perform_chef_augmentation_analysis.py
#
# Train model using the CHEF data
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
gpu_id = sys.argv[4]

file_folder = "../process_data/chef/output/phylo_avg_data/chef_and_clef/"
homolog_folder = "../process_data/chef/output/phylo_avg_data/chef_and_clef/orthologs_6/"
#output_folder = "./output_chef_one_homolog_20_species/"
output_folder = "./output_chef_one_homolog_6_species_linear/"
tasks = ['h3k27ac_log2_enrichment', 'tf_sum']
sequence_size = 700
sample_fraction = 1.0

# ====================================================================================================================
# Main code
# ====================================================================================================================
if model_type == "deepstarr":
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, tasks, sequence_size, model_type, gpu_id)
elif model_type == "explainn":
    models.train_explainn(use_homologs, sample_fraction, replicate, file_folder,
                          homolog_folder, output_folder, tasks, sequence_size, model_type, gpu_id)
elif model_type == "motif_deepstarr":
    models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                                 homolog_folder, output_folder, tasks, sequence_size, model_type, gpu_id)
elif model_type == "db_linear":
    models.train_db_linear(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, tasks, sequence_size, model_type, gpu_id)
