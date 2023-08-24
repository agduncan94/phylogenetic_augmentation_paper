#!/bin/bash

# ####################################################################################################################
# perform_drosophila_sampling_analysis.sh
#
# Run the DeepSTARR model on sampled training data
# ####################################################################################################################

num_replicates=1
declare -a models=('deepstarr')
declare -a sample_fractions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for model in "${models[@]}"
do
	for frac in ${sample_fractions[@]}
	do
		for replicate in $(seq 1 $num_replicates)
		do
			python perform_drosophila_sampling_analysis.py "${model}" ${frac} 0 ${replicate}
			
			python perform_drosophila_sampling_analysis.py "${model}" ${frac} 1 ${replicate}
		done
	done
done
