#!/bin/bash

# ####################################################################################################################
# perform_mesc_classification_analysis.sh
#
# Run each model on the same mesc data. Three replicates per model.
# ####################################################################################################################

num_replicates=1
#declare -a models=('motif_deepstarr' 'deepstarr' 'motif_linear' 'explainn')
declare -a models=('motif_linear')

gpu_id=$1

for m in "${models[@]}"
do
	for i in $(seq 1 $num_replicates)
	do
		echo "Training ${m} model - replicate ${i} - with homologs"
		python perform_mesc_classification_analysis.py "${m}" ${i} 1 ${gpu_id}
		
		echo "Training ${m} model - replicate ${i} - without homologs"
		python perform_mesc_classification_analysis.py "${m}" ${i} 0 ${gpu_id}
		
	done
done
