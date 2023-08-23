# ####################################################################################################################
# perform_drosophila_num_species_analysis.py
#
# Train model using the STARR-seq data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
import models as models
import pandas as pd

# ====================================================================================================================
# Arguments
# ====================================================================================================================
model_type = sys.argv[1]
replicate = sys.argv[2]
use_homologs = bool(int(sys.argv[3]))
num_species = int(sys.argv[4])
gpu_id = sys.argv[5]

file_folder = "../process_data/drosophila/output/"
homolog_folder = "../process_data/drosophila/output/orthologs/"
output_folder = "./output_drosophila_num_species_rev/"
tasks = ['Dev', 'Hk']
sequence_size = 249
sample_fraction = 1.0
species_file = "../process_data/drosophila/output/ordered_drosophila_species.txt"

# ====================================================================================================================
# Main code
# ====================================================================================================================

# Read species file
species_df = pd.read_csv(species_file, sep='\t')
species_list = species_df['species'].to_list()

species_list.reverse()

species_list = species_list[0:num_species]


if model_type == "deepstarr":
    models.train_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                           homolog_folder, output_folder, tasks, sequence_size, species_list, None, model_type + '_' + str(num_species), gpu_id)
elif model_type == "explainn":
    models.train_explainn(use_homologs, sample_fraction, replicate, file_folder,
                          homolog_folder, output_folder, tasks, sequence_size, species_list, None, model_type + '_' + str(num_species), gpu_id)
elif model_type == "motif_deepstarr":
    models.train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder,
                                 homolog_folder, output_folder, tasks, sequence_size, species_list, None, model_type + '_' + str(num_species), gpu_id)
