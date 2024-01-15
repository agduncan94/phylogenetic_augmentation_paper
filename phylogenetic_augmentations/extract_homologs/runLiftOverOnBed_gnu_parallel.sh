#!/bin/bash

# Run HAL liftover on a set of sequences from one species to another from the alignment file
# =============================================================================================
# ARGUMENTS:
# $1 - BED file
# $2 - Name of query species (use name from alignment) 
# $3 - Path to file containing list of target species names
# $4 - Prefix for output files
# $5 - Folder to place output in
# $6 - Size to resize sequences to

bed_file=$1
query_species=$2
species_file=$3
output_name=$4
output_folder=$5
sequence_size=$6

bed_name=$(basename ${bed_file})

# GLOBAL VARIABLES:
hal_file="drosophila.hal"
halper_location="./halLiftover-postprocessing"
fasta_folder="./fastas"
num_processes=5

# MAIN CODE:
# Create output folders
mkdir -p ${output_folder}/orthologs
mkdir -p ${output_folder}/orthologs/bed
mkdir -p ${output_folder}/orthologs/per_species_fa

# Create summit peak file
awk '{summit=(int($2)+int($3))/2;printf("%s\t%d\t%d\t%s\n",$1,summit,summit+1,$4)}' $bed_file > ${output_folder}/orthologs/bed/${bed_name}.summit.bed

# Create query species FASTA
bedtools getfasta -fi ${fasta_folder}/${query_species}.fa -bed ${bed_file} -fo ${output_folder}/orthologs/per_species_fa/${output_name}.${query_species}.fa -name

# Run HAL liftover on multiple species in parallel
cat ${species_file} | parallel -j ${num_processes}  "bash ./runLiftOverOnBed_helper.sh ${bed_file} ${query_species} {} ${output_name} ${output_folder} ${sequence_size}"

