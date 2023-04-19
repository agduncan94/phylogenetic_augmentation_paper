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
elif model_type == "simple_cnn":
	models.train_simple_model(use_homologs, replicate, model_type)
elif model_type == "motif_db_linear":
	models.train_motif_db_linear_model(use_homologs, replicate, model_type)	
elif model_type == "motif_db_deepstarr":
	models.train_motif_db_deepstarr_model(use_homologs, replicate, model_type)	
elif model_type == "motif_learned_deepstarr":
	models.train_motif_learned_deepstarr_model(use_homologs, replicate, model_type)	