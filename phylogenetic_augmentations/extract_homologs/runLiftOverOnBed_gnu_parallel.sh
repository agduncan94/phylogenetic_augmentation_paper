#!/bin/bash

# Run HAL liftover on a set of sequences from one species to another from Zoonomia
bed_file=$1
query_species=$2
species_file=$3
output_name=$4
output_folder=$5
sequence_size=$6

# Make output folder
mkdir -p ${output_folder}/orthologs
mkdir -p ${output_folder}/orthologs/bed
mkdir -p ${output_folder}/orthologs/per_species_fa

bed_name=$(basename ${bed_file})

hal_file="drosophila.hal"
halper_location="./halLiftover-postprocessing"
fasta_folder="./fastas"

# Create summit peak file
awk '{summit=(int($2)+int($3))/2;printf("%s\t%d\t%d\t%s\n",$1,summit,summit+1,$4)}' $bed_file > ${output_folder}/orthologs/bed/${bed_name}.summit.bed

# Create query species FASTA
bedtools getfasta -fi ${fasta_folder}/${query_species}.fa -bed ${bed_file} -fo ${output_folder}/orthologs/per_species_fa/${output_name}.${query_species}.fa -name

# Run in parallel
cat ${species_file} | parallel -j 5  "bash ./runLiftOverOnBed_helper.sh ${bed_file} ${query_species} {} ${output_name} ${output_folder} ${sequence_size}"

