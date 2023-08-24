# ####################################################################################################################
# perform_basset_analysis.py
#
# Train model using the Basset data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import numpy as np
import utils
import models_basset as models

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))
sample_fraction = float(sys.argv[4])

file_folder = "../process_data/basset/output/"
homolog_folder = "../process_data/basset/output/orthologs/per_species_fa/"
output_folder = "./output_basset_sampling/"

# ====================================================================================================================
# Main code
# ====================================================================================================================

num_samples_train = utils.count_lines_in_file(
    file_folder + "Sequences_activity_Train.txt") - 1

filtered_indices = None
if int(sample_fraction) < 1:
    reduced_num_samples_train = int(num_samples_train * sample_fraction)
    filtered_indices = np.random.choice(
        list(range(num_samples_train)), reduced_num_samples_train, replace=False)

models.train_basset(use_homologs, sample_fraction, replicate, file_folder,
                    homolog_folder, output_folder, filtered_indices)

models.fine_tune_basset(use_homologs, sample_fraction, replicate, file_folder,
                        homolog_folder, output_folder, filtered_indices)
