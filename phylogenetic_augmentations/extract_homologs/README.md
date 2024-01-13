# Steps for extracting homologs from a multi-species genome alignment

This folder contains scripts to extract homologs for a set of sequences from a multi-species genome alignment.

The code shows an example of how to extract homologs for *Drosophila* S2 STARR-seq sequences.
We hope it serves as an example for users who want to extract homologs for their own genomic sequences.

## Setting up the environment
The environment requires a few dependencies, including:

* python >=3.6
* hdf5
* numpy and matplotlib
* [hal](https://github.com/ComparativeGenomicsToolkit/hal)
* [HALPER](https://github.com/pfenninglab/halLiftover-postprocessing)

We recommend following the HALPER documentation to install hal and HALPER:
[Installation](https://github.com/pfenninglab/halLiftover-postprocessing/blob/master/hal_install_instructions.md)

## Downloading the files
The following input files are required:

* Cactus alignment: [Drosophila](https://github.com/flyseq/2023_drosophila_assembly) -> drosophila.hal
* Drosophila S2 bed file: [Zenodo link]() -> drosophila_train.dm6.bed

Put these into the current working directory.

Note: When extracting homologs from an alignment, make sure that the bed file is in the correct assembly version.

## Extract FASTA files and chromosome size files
In order to identify homologs, you will need the FASTA file for each species in the alignment that you plan to call homologs for, along with a chromosome size file.

The following code will create a folder with these files.

`
# Create a file with a list of species of interest
# Use halStats --genomes <path/to/drosophila.hal> and filter the results
# Store into ./drosophila_species_file.txt
# There is a sample file as an example

# Get FASTAs for species in list from the drosophila.hal file
bash getFastasFromList.sh drosophila.hal drosophila_species_file.txt
`

## Call homologs on Drosophila S2 training sequences

Before running, please see the following notes:
* This code assumes that halper is installed in the current directory. If it is installed elsewhere, please update `runLiftOverOnBed_gnu_parallel.sh` and `runLiftOverOnBed_helper.sh` accordingly.
* The script `runLiftOverOnBed_gnu_parallel.sh` extracts 5 homologs at a time. This can be changed based on available IO by editing the file.
* We recommend using nohup and &, as running for all species take a few days

`
# Run liftover
nohup bash runLiftOverOnBed_gnu_parallel.sh drosophila_s2_starr_seq_train.dm6.bed D_MELANOGASTER drosophila_species_file.txt drosophila_s2_train ./drosophila_s2_train 249 &
`

As the code performs liftOver on species, you can find the homolog FASTA files in `./drosophila_s2_train/per_species_fa/`.

The name of each sequence in each FASTA is of the form <original sequence name>::<homolog species coordinates>