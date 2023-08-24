#!/bin/bash

# ####################################################################################################################
# perform_drosophila_homolog_rate_analysis.sh
#
# Run each model on the same STARR-seq data. Three replicates per model.
# ####################################################################################################################

num_replicates=2
declare -a models=('deepstarr')
declare -a homolog_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for model in "${models[@]}"
do
    for homolog_rate in ${homolog_rates[@]}
    do
    	for replicate in $(seq 1 $num_replicates)
    	do
    		python perform_drosophila_homolog_rate_analysis.py "${model}" ${replicate} 1 ${homolog_rate}
    		
    	done
    done
done

for model in "${models[@]}"
do
   	for replicate in $(seq 1 $num_replicates)
   	do
   		python perform_drosophila_homolog_rate_analysis.py "${model}" ${replicate} 0 1
   		
   	done
done