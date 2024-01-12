# Steps for recreating the analysis figures from the paper

This folder contains all of the scripts necessary to recreate the phylogenetic augmentation analysis and associated figures.

Note that the runtime of these analyses is dependent on the system being used. All analyses were run on an Ubuntu workstation with the following specs:

* Ubuntu 20.04.6 LTS
* Intel Core i9-10900X CPU @ 3.70GHz x 20
* NVIDIA RTX A400 GPU with 16376MiB of memory
* 31.1 GiB of memory

## Download data

The data for the analyses are stored on Zenodo and can be downloaded from there.

See `../input/README.md` for instructions.

## Initialize the conda environment

Loading from the conda file should create the same environment that was used to run the initial analysis. The analysis requires access to a GPU.

Requires R and Conda.

```
# Create the conda environment from file
conda create --name phylogenetic_augmentations python=3.8 cudatoolkit=11.2.2

# Activate the conda env
conda activate phylogenetic_augmentations

# Install libraries via pip
pip install -r requirements.txt

# Install the R pacakges:
# * tidyverse
# * cowplot

```

## Perform model analysis (Figure 2)

To run the analysis for different models, run the following commands:

```
# Train three different CNN models on Drosophila S2 data
python perform_drosophila_augmentation_analysis.py

# Train the Basset CNN on the Basset data
python perform_basset_augmentation_analysis.py

# Rename and copy model_metrics.tsv files to '../output/'
cp ../output/drosophila_augmentation/model_metrics.tsv ../output/drosophila_augmentation_metrics.tsv
cp ../output/basset_augmentation/model_metrics.tsv ../output/basset_augmentation_metrics.tsv

# Create final figures
Rscript --vanilla plot_phylo_aug_model_results.R
```

## Perform sampling analysis and 3' UTR analysis (Figure 3 and Supplemental Figure 1)

To run the sampling analysis and 3'UTR analysis, run the following commands:

```
# Train models on sampled Drosophila S2 data
python perform_drosophila_sampling_analysis.py

# Train models on sampled Basset data
python perform_basset_sampling_analysis.py

# Train models on the yeast data
python perform_utr_augmentation_analysis.py

# Rename and copy model_metrics.tsv files to '../output/'
cp ../output/drosophila_sampling/model_metrics.tsv ../output/drosophila_sampling_metrics.tsv
cp ../output/basset_sampling/model_metrics.tsv ../output/basset_sampling_metrics.tsv
cp ../output/yeast_augmentation/model_metrics.tsv ../output/yeast_model_metrics.tsv

# Perform global importance analysis
python extract_learned_motifs.py
cp ../output/puf3_motif_importance/PUF3_predicted_binding_baseline.tsv ../output/PUF3_predicted_binding_baseline.tsv
cp ../output/puf3_motif_importance/PUF3_predicted_binding_augmented.tsv ../output/PUF3_predicted_binding_augmented.tsv

# Create final figures
Rscript --vanilla plot_phylo_aug_sampling_analysis.R
Rscript --vanilla plot_yeast_3utr_motif_analysis.R
```

## Perform hyperparameter analysis (Figure 4 and Supplemental Figures 2 and 3)

To run the hyperparameter analysis, run the following commands:

```
# Train models for homolog rate with 137 species
python perform_drosophila_homolog_rate_analysis.py

# Train models for number of species
python perform_drosophila_num_species_analysis.py

# Train models for homolog rate with first 10 species
python perform_drosophila_phylo_rate_fewer_species.py

# Train models for number of species (reverse)
python perform_drosophila_num_species_analysis_reverse.py

# Rename and copy model_metrics.tsv files to '../output/'
cp ../output/drosophila_phylo_aug_rate/model_metrics.tsv ../output/drosophila_phylo_aug_rate_metrics.tsv
cp ../output/drosophila_num_species/model_metrics.tsv ../output/drosophila_num_species_metrics.tsv
cp ../output/drosophila_phylo_aug_rate_reduced_species/model_metrics.tsv ../output/drosophila_phylo_aug_rate_reduced_species_metrics.tsv
cp ../output/drosophila_num_species_rev/model_metrics.tsv ../output/drosophila_num_species_rev_metrics.tsv

# Create final figures
Rscript --vanilla plot_hyperparameter_analysis.R
Rscript --vanilla plot_hyperparameter_analysis_suppl_phylo_aug_rate.R
Rscript --vanilla plot_hyperparameter_analysis_suppl_num_species_rev.R
```

# Misc

## Format of the HDF5 file

For the BASSET data, the training data (including homologs) was too large to load into memory all at once. Instead, the sequences were stored into an HDF5 file with the following structure.

There were three groups: `Training`, `Validation` and `Testing`. Each group contained a subgroup for `sequences`.
The `sequences` contained a dataset for each sequence from the corresponding set. For the `Training` sequences, this dataset included all homologs for that sequence.
