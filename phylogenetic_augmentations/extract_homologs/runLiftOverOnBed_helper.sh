#!/bin/bash

# Run HAL liftover on a set of sequences from one species to another from Zoonomia
# HELPER SCRIPT
bed_file=$1
query_species=$2
target_species=$3
output_name=$4
output_folder=$5
sequence_size=$6

bed_name=$(basename ${bed_file})

hal_file="drosophila.hal"
halper_location="./halLiftover-postprocessing"
fasta_folder="./fastas"

# For each species, perform liftOver
echo "Liftover of BED file"
# Perform liftover on bed file
halLiftover ${hal_file} ${query_species} ${bed_file} ${target_species} ${output_folder}/orthologs/bed/${output_name}.${target_species}.full.liftover.bed

echo "Liftover of summit file"
# Perform liftover on summits
halLiftover ${hal_file} ${query_species} ${output_folder}/orthologs/bed/${bed_name}.summit.bed ${target_species} ${output_folder}/orthologs/bed/${output_name}.${target_species}.summit.liftover.bed

echo "Running orthologFind"
# Run HALPER
python ${halper_location}/orthologFind.py -max_frac 1.25 -min_frac 0.1 -qFile ${bed_file} -tFile ${output_folder}/orthologs/bed/${output_name}.${target_species}.full.liftover.bed -sFile ${output_folder}/orthologs/bed/${output_name}.${target_species}.summit.liftover.bed -oFile ${output_folder}/orthologs/bed/${output_name}.${target_species}.bed -mult_keepone -protect_dist 5 -narrowPeak 

echo "Resizing to ${sequence_size} bp"
# Resize to new size
bash ./extendSimpleBedByPeak.sh ${output_folder}/orthologs/bed/${output_name}.${target_species}.bed ${sequence_size} ${fasta_folder}/${target_species}.fa.fai.chrom.sizes

echo "Converting to FASTA"
# Convert to FASTA
bedtools getfasta -fi ${fasta_folder}/${target_species}.fa -bed ${output_folder}/orthologs/bed/${output_name}.${target_species}.bed_resized.bed -fo ${output_folder}/orthologs/per_species_fa/${output_name}.${target_species}.fa -name

echo "Clean up"
rm ${output_folder}/orthologs/bed/${output_name}.${target_species}.full.liftover.bed
rm ${output_folder}/orthologs/bed/${output_name}.${target_species}.summit.liftover.bed
