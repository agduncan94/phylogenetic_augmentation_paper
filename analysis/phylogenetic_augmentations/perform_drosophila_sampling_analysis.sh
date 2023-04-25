#!/bin/bash

# ####################################################################################################################
# perform_drosophila_sampling_analysis.sh
#
# Run the DeepSTARR model on sampled training data
# ####################################################################################################################

num_replicates=3
declare -a models=('deepstarr')
declare -a sample_fraction=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for m in "${models[@]}"
do
	for i in ${sample_fraction[@]}
	do
		for n in $(seq 1 $num_replicates)
		do
			echo "Training ${m} model - replicate ${n} - percent ${i} - without homologs"
			python perform_drosophila_sampling_analysis.py "${m}" ${n} 0 ${i}
			
			echo "Training ${m} model - replicate ${n} - percent ${i} - with homologs"
			python perform_drosophila_sampling_analysis.py "${m}" ${n} 1 ${i}
		done
	done
done
