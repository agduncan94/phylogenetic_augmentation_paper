#!/bin/bash

# Run HAL liftover on a set of sequences from one species to another from the alignment file
# THIS IS A HELPER SCRIPT THAT IS NOT MEANT TO BE RUN DIRECTLY
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
target_species=$3
output_name=$4
output_folder=$5
sequence_size=$6

bed_name=$(basename ${bed_file})

# GLOBAL VARIABLES:
hal_file="drosophila.hal"
halper_location="./halLiftover-postprocessing"
fasta_folder="./fastas"

# MAIN CODE:

# Perform liftover on bed file
echo "Liftover of BED file"
halLiftover ${hal_file} ${query_species} ${bed_file} ${target_species} ${output_folder}/orthologs/bed/${output_name}.${target_species}.full.liftover.bed

# Perform liftover on summits
echo "Liftover of summit file"
halLiftover ${hal_file} ${query_species} ${output_folder}/orthologs/bed/${bed_name}.summit.bed ${target_species} ${output_folder}/orthologs/bed/${output_name}.${target_species}.summit.liftover.bed

# Run HALPER to find homologs
echo "Running orthologFind"
python ${halper_location}/orthologFind.py -max_frac 1.25 -min_frac 0.1 -qFile ${bed_file} -tFile ${output_folder}/orthologs/bed/${output_name}.${target_species}.full.liftover.bed -sFile ${output_folder}/orthologs/bed/${output_name}.${target_species}.summit.liftover.bed -oFile ${output_folder}/orthologs/bed/${output_name}.${target_species}.bed -mult_keepone -protect_dist 5 -narrowPeak 

# Resize to homologs to new size
echo "Resizing to ${sequence_size} bp"
bash ./extendSimpleBedByPeak.sh ${output_folder}/orthologs/bed/${output_name}.${target_species}.bed ${sequence_size} ${fasta_folder}/${target_species}.fa.fai.chrom.sizes

# Convert to FASTA
echo "Converting to FASTA"
bedtools getfasta -fi ${fasta_folder}/${target_species}.fa -bed ${output_folder}/orthologs/bed/${output_name}.${target_species}.bed_resized.bed -fo ${output_folder}/orthologs/per_species_fa/${output_name}.${target_species}.fa -name

# Remove extra files
echo "Clean up"
rm ${output_folder}/orthologs/bed/${output_name}.${target_species}.full.liftover.bed
rm ${output_folder}/orthologs/bed/${output_name}.${target_species}.summit.liftover.bed
