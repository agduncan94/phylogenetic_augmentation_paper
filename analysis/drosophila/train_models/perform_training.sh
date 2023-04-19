#!/bin/bash

# ####################################################################################################################
# perform_training.sh
#
# Run each model on the same STARR-seq data. Four replicates per model.
# ####################################################################################################################

num_replicates=3
#declare -a models=('explainn' 'simple_cnn' 'deepstarr')
declare -a models=('motif_db_linear' 'motif_db_deepstarr' 'motif_learned_deepstarr')

for m in "${models[@]}"
do
	for i in $(seq 1 $num_replicates)
	do
		echo "Training ${m} model - replicate ${i} - without homologs"
		python perform_training.py "${m}" ${i} 0
		
		echo "Training ${m} model - replicate ${i} - with homologs"
		python perform_training.py "${m}" ${i} 1
	done
done
