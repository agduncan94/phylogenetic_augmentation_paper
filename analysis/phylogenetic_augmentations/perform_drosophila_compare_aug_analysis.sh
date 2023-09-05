#!/bin/bash

# ####################################################################################################################
# perform_drosophila_compare_aug_analysis.sh
#
# Run each model on the same STARR-seq data with different homologs. Three replicates per model.
# ####################################################################################################################

num_replicates=1
declare -a models=('deepstarr')

gpu_id=$1

for m in "${models[@]}"
do
	for i in $(seq 1 $num_replicates)
	do
		
		#echo "Training ${m} model - replicate ${i} - with homologs"
		#python perform_drosophila_compare_aug_analysis.py "${m}" ${i} 1 ${gpu_id}
		
		echo "Training ${m} model - replicate ${i} - without homologs"
		python perform_drosophila_compare_aug_analysis.py "${m}" ${i} 0 ${gpu_id}
		
	done
done