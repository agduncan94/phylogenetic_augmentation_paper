# ####################################################################################################################
# perform_mesc_classification_analysis.py
#
# Train model using the mesc data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models_binary_classification as models

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))
gpu_id = sys.argv[4]

file_folder = "../process_data/chef/output_chef_clef_classification/"
homolog_folder = "../process_data/chef/output_chef_clef_classification/orthologs/"
output_folder = "./output_chef_clef_classification/"
sequence_size = 700
sample_fraction = 1.0

# ====================================================================================================================
# Main code
# ====================================================================================================================
if model_type == "deepstarr":
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, sequence_size, None, None, model_type, gpu_id)
elif model_type == "explainn":
    models.train_explainn(use_homologs, sample_fraction, replicate, file_folder,
                          homolog_folder, output_folder, sequence_size, None, None, model_type, gpu_id)
elif model_type == "motif_deepstarr":
    models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                                 homolog_folder, output_folder, sequence_size, None, None, model_type, gpu_id)
elif model_type == "motif_linear":
    models.train_motif_linear(use_homologs, sample_fraction, replicate, file_folder,
                              homolog_folder, output_folder, sequence_size, None, None, model_type, gpu_id)
elif model_type == "motif_linear_relu":
    models.train_motif_linear_relu(use_homologs, sample_fraction, replicate, file_folder,
                                   homolog_folder, output_folder, sequence_size, None, None, model_type, gpu_id)
