#!/bin/bash

# ####################################################################################################################
# perform_drosophila_num_species_analysis.sh
#
# Run each model on the same STARR-seq data. Three replicates per model.
# ####################################################################################################################

num_replicates=3
declare -a models=('deepstarr')

gpu_id=$1

num_species=20

for m in "${models[@]}"
do
	for i in $(seq 1 $num_replicates)
	do
    	for s in $(seq 1 $num_species)
		do
    		echo "Training ${m} model - replicate ${i} - with homologs (${s} species)"
    		python perform_drosophila_num_species_analysis.py "${m}" ${i} 1 ${s} ${gpu_id}
		done
	done
done
