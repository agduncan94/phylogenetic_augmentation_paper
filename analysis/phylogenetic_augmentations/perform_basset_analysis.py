# ####################################################################################################################
# perform_basset_analysis.py
#
# Train model using the Basset data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models_basset as models

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))
sample_fraction = float(sys.argv[4])
gpu_id = sys.argv[5]

file_folder = "../process_data/basset/output/"
homolog_folder = "../process_data/basset/output/orthologs/per_species_fa/"
output_folder = "./output_basset_test_hdf5/"
sequence_size = 600
tasks = list(map(str, list(range(0, 164))))


# ====================================================================================================================
# Main code
# ====================================================================================================================
models.train_basset(use_homologs, sample_fraction, replicate, file_folder,
                    homolog_folder, output_folder, tasks, sequence_size, model_type, gpu_id)
