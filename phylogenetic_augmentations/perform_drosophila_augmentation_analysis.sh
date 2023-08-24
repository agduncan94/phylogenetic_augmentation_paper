#!/bin/bash

# ####################################################################################################################
# perform_drosophila_augmentation_analysis.sh
#
# Run each model on the same STARR-seq data. Three replicates per model.
# ####################################################################################################################

num_replicates=1
#declare -a models=('deepstarr' 'motif_deepstarr' 'explainn')
declare -a models=('deepstarr')

for model in "${models[@]}"
do
	for replicate in $(seq 1 $num_replicates)
	do
		# Train model without phylogenetic augmentation
		python perform_drosophila_augmentation_analysis.py "${model}" ${replicate} 0
		
		# Train model with phylogenetic augmentation
		python perform_drosophila_augmentation_analysis.py "${model}" ${replicate} 1
		
	done
done
