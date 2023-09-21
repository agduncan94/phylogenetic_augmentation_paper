# ####################################################################################################################
# perform_basset_sampling_analysis.py
#
# Train model on the Basset data with varying fractions of training data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import numpy as np
import utils
import models_basset as models

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
model_types = ['basset']
num_replicates = 3
fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

file_folder = "../basset/"
output_folder = "../output/basset_sampling/"

# ====================================================================================================================
# Main code
# ====================================================================================================================


def train_model(use_homologs, model_type, replicate, fraction, filtered_indices):
    models.train_basset(use_homologs, fraction, replicate, file_folder,
                        output_folder, filtered_indices=filtered_indices)

    models.fine_tune_basset(use_homologs, fraction, replicate, file_folder,
                            output_folder, filtered_indices=filtered_indices)


for model_type in model_types:
    for fraction in fractions:
        for replicate in range(1, num_replicates + 1):
            num_samples_train = utils.count_lines_in_file(
                file_folder + "Sequences_Train.txt") - 1

            filtered_indices = None
            if int(fraction) < 1:
                reduced_num_samples_train = int(
                    num_samples_train * fraction)
                filtered_indices = np.random.choice(
                    list(range(num_samples_train)), reduced_num_samples_train, replace=False)

            train_model(True, model_type, replicate,
                        fraction, filtered_indices)
            train_model(False, model_type, replicate,
                        fraction, filtered_indices)
