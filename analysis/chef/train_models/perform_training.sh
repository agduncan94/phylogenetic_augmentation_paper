#!/bin/bash

# ####################################################################################################################
# perform_training.sh
#
# Run each model on the same CHEF data. Four replicates per model.
# ####################################################################################################################

#num_replicates=3
#declare -a models=('simple_cnn' 'deepstarr' 'explainn')

num_replicates=1
declare -a models=('deepstarr')

for m in "${models[@]}"
do
	for i in $(seq 1 $num_replicates)
	do
		echo "Training ${m} model - replicate ${i} - with homologs"
		python perform_training.py "${m}" ${i} 1
		
		echo "Training ${m} model - replicate ${i} - without homologs"
		python perform_training.py "${m}" ${i} 0
	done
done
