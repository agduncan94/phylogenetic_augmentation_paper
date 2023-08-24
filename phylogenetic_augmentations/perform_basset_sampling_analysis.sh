#!/bin/bash

# ####################################################################################################################
# perform_basset_analysis.sh
#
# Run each model on the same basset. Three replicates per model.
# ####################################################################################################################

num_replicates=3
declare -a models=('basset')
declare -a sample_fraction=(0.1 0.2 0.3 0.4 0.5 1.0)

for model in "${models[@]}"
do
    for frac in ${sample_fraction[@]}
	do
    	for replicate in $(seq 1 $num_replicates)
    	do       	
    		python perform_basset_analysis.py "${model}" ${replicate} 0 ${frac}
    		
            python perform_basset_analysis.py "${model}" ${replicate} 1 ${frac}
		done
	done
done
