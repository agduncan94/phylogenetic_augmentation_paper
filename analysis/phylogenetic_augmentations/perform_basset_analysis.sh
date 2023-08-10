#!/bin/bash

# ####################################################################################################################
# perform_basset_analysis.sh
#
# Run each model on the same basset. Three replicates per model.
# ####################################################################################################################

num_replicates=3
declare -a models=('basset')
declare -a sample_fraction=(0.1 0.2 0.3 0.4 0.5 1.0)

gpu_id=$1

for m in "${models[@]}"
do
    for i in ${sample_fraction[@]}
	do
    	for n in $(seq 1 $num_replicates)
    	do       	
    		echo "Training ${m} model - replicate ${n} - sample ${i} - without homologs"
    		python perform_basset_analysis.py "${m}" ${n} 0 ${i} ${gpu_id}
    		
    		#echo "Training ${m} model - replicate ${n} - sample ${i} - with homologs"
            #python perform_basset_analysis.py "${m}" ${n} 1 ${i} ${gpu_id}
		done
	done
done
