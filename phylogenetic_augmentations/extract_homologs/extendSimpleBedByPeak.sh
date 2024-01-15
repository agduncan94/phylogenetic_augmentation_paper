#!/bin/bash

# Extend a bed file by finding the midpoint extending on both sides by n
# $1 - bed file
# $2 - new length of sequence
# $3 - reference genome file (chromosome sizes file)

# Find the new midpoint of the sequence
awk '{midpoint=(int($2)+int($3))/2;printf("%s\t%d\t%d\t%s\n",$1,midpoint,midpoint,$4)}' $1  > ${1}_bed.midpoint.bed

# Extend the sequence by n
bedtools slop -i ${1}_bed.midpoint.bed -g $3 -l $(($2/2)) -r $((($2+1)/2)) > ${1}_resized.bed
rm ${1}_bed.midpoint.bed
