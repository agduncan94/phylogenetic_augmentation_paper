#!/bin/bash

# ####################################################################################################################
# perform_chef_augmentation_analysis.sh
#
# Run each model on the same CHEF data. Three replicates per model.
# ####################################################################################################################

num_replicates=3
#declare -a models=('deepstarr' 'explainn' 'motif_deepstarr')
declare -a models=('deepstarr')
gpu_id=$1

for m in "${models[@]}"
do
	for i in $(seq 1 $num_replicates)
	do
		echo "Training ${m} model - replicate ${i} - without homologs"
		python perform_chef_augmentation_analysis.py "${m}" ${i} 0 ${gpu_id}
		
		echo "Training ${m} model - replicate ${i} - with homologs"
		python perform_chef_augmentation_analysis.py "${m}" ${i} 1 ${gpu_id}
	done
done
