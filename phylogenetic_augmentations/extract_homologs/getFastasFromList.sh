#!/bin/bash

# Get FASTA files from a hal alignment
# $1 - hal alignment file (e.g., drosophila.hal)
# $2 - species file - each line should have a species (using nomenclature from hal alignment)

#hal_alignment="/neuhaus/andrew/drosophila/drosophila.hal"
#species_file="../drosophila.genomes.txt"
hal_alignment=$1
species_file=$2
while read species; do
  echo $species
  hal2fasta $hal_alignment $species --outFaPath ${species}.fa
done <$species_file