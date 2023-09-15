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

```
# Load the conda environment from file
conda create --name phylogenetic_augmentations python=3.8 cudatoolkit=11.2.2

# Activate the conda env
conda activate phylogenetic_augmentations

# Install libraries via pip
pip install -r requirements.txt

```

## Optional: Run script on a small example set

The following optional code will run phylogenetic augmentations on a small example set (10% of the drosophila data) to validate everything is installed and connected:

```
# Train a DeepSTARR model on test data
python perform_drosophila_test.py

# Create a figure
Rscript --vanilla plot_drosophila_test.R

```

## Perform model analysis (Figure 1)

To run the analysis for different models, run the following commands:

```
# Train three different CNN models on Drosophila S2 data
python perform_drosophila_augmentation_analysis.py

# Train the Basset CNN on the Basset data
python perform_basset_augmentation_analysis.py

# Create final figures
Rscript --vanilla plot_phylo_aug_model_results.R
```

## Perform sampling analysis (Figure 2)

To run the sampling analysis, run the following commands:

```
# Train models on sampled Drosophila S2 data
python perform_drosophila_sampling_analysis.py

# Train models on sampled Basset data
python perform_basset_sampling_analysis.py

# Create final figures
Rscript --vanilla plot_phylo_aug_sampling_analysis.R
```

## Perform hyperparameter analysis (Figure 3)

To run the hyperparameter analysis, run the following commands:

```
# Train models for homolog rate
python perform_drosophila_homolog_rate_analysis.py

# Train models for number of species
python perform_drosophila_num_species_analysis.py

# Create final figures
Rscript --vanilla plot_hyperparameter_analysis.R
```

# Misc

## Format of the HDF5 file

For the BASSET data, the training data (including homologs) was too large to load into memory all at once. Instead, the sequences were stored into an HDF5 file with the following structure.

There were three groups: `Training`, `Validation` and `Testing`. Each group contained a subgroup for `sequences`.
The `sequences` contained a dataset for each sequence from the corresponding set. For the `Training` sequences, this dataset included all homologs for that sequence.
