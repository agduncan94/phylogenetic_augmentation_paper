#!/bin/bash

# ####################################################################################################################
# perform_training.sh
#
# Run each model on the same STARR-seq data. Four replicates per model.
# ####################################################################################################################

num_replicates=1
#declare -a models=('motif_deepstarr' 'explainn' 'deepstarr' 'motif_avg' 'motif_max')
declare -a models=('motif_deepstarr')

for m in "${models[@]}"
do
	for i in $(seq 1 $num_replicates)
	do		
		echo "Training ${m} model - replicate ${i} - with homologs"
		python perform_training.py "${m}" ${i} 1
	done
done
