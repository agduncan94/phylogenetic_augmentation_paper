#!/bin/bash

#file_folder="/home/andrew/Documents/research/phd_research/deep_learning/evolutionary_augmentations/Drosophila/add_homologs/input/168_way_249bp/"
#homolog_dir="/home/andrew/Documents/research/phd_research/deep_learning/evolutionary_augmentations/Drosophila/add_homologs/input/168_way_249bp/orthologs/per_species_fa/"
file_folder="../input/168_way_drosophila_msa/"
homolog_dir="../input/168_way_drosophila_msa/orthologs/"

output_folder="../output/evolutionary_augmentations/"
mkdir -p $output_folder

rm "${output_folder}/model_scc.tsv"
touch "${output_folder}/model_scc.tsv"
echo -e 'dev_train\thk_train\tdev_val\thk_val\tdev_test\thk_test\tepoch\ttype\tmodel\tattempt' > "${output_folder}/model_scc.tsv"

python train_models_with_evolutionary_augmentations_simple_cnn.py $file_folder $homolog_dir $output_folder 4
python train_models_with_evolutionary_augmentations_explainn.py $file_folder $homolog_dir $output_folder 4
python train_models_with_evolutionary_augmentations_deepstarr.py $file_folder $homolog_dir $output_folder 4

