#!/bin/bash

# ####################################################################################################################
# perform_drosophila_homolog_rate_analysis.sh
#
# Run each model on the same STARR-seq data. Three replicates per model.
# ####################################################################################################################

num_replicates=2
declare -a models=('deepstarr')
declare -a homolog_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

gpu_id=$1

for m in "${models[@]}"
do
    for r in ${homolog_rates[@]}
    do
    	for i in $(seq 1 $num_replicates)
    	do
    		echo "Training ${m} model - replicate ${i} - homolog rate ${r} - with homologs"
    		python perform_drosophila_homolog_rate_analysis.py "${m}" ${i} 1 ${r} ${gpu_id}
    		
    	done
    done
done

for m in "${models[@]}"
do
   	for i in $(seq 1 $num_replicates)
   	do
   		echo "Training ${m} model - replicate ${i} - homolog rate ${r} - without homologs"
   		python perform_drosophila_homolog_rate_analysis.py "${m}" ${i} 0 1 ${gpu_id}
   		
   	done
done