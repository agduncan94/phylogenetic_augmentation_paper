# ####################################################################################################################
# perform_training.py
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

# ====================================================================================================================
# Main code
# ====================================================================================================================
if model_type == "deepstarr":
	models.train_deepstarr(use_homologs, replicate, model_type)
elif model_type == "explainn":
	models.train_explainn(use_homologs, replicate, model_type)
elif model_type == "motif_avg":
	models.train_motif_avg_model(use_homologs, replicate, model_type)
elif model_type == "motif_max":
	models.train_motif_max_model(use_homologs, replicate, model_type)
elif model_type == "motif_deepstarr":
	models.train_motif_deepstarr(use_homologs, replicate, model_type)