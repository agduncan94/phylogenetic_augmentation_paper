# ####################################################################################################################
# perform_drosophila_compare_aug_analysis.py
#
# Train model using the STARR-seq data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models as models
from itertools import chain, combinations

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))
gpu_id = sys.argv[4]

file_folder = "../process_data/drosophila/output/"
homolog_folder = "../process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila_augs_compare/"
tasks = ['Dev', 'Hk']
sequence_size = 249
sample_fraction = 1.0

sequence_filters = ['peak_849bp_region', 'Other']


# ====================================================================================================================
# Main code
# ====================================================================================================================


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


for i, combo in enumerate(powerset(sequence_filters), 1):
    full_combo = list(combo)
    full_combo.append('positive_peaks')
    full_combo.append('negative')
    print(full_combo)
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, tasks, sequence_size, None, full_combo, model_type, gpu_id)
