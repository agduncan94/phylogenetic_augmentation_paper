#!/bin/bash

# ####################################################################################################################
# perform_drosophila_num_species_analysis.sh
#
# Run each model on the same STARR-seq data. Three replicates per model.
# ####################################################################################################################

num_replicates=3
declare -a models=('deepstarr')

num_species=20

for model in "${models[@]}"
do
	for replicate in $(seq 1 $num_replicates)
	do
    	for species in $(seq 1 $num_species)
		do
    		python perform_drosophila_num_species_analysis.py "${model}" ${replicate} 1 ${species}
		done
	done
done
