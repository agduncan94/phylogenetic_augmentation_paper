# Steps for recreating the analysis figures from the paper

This folder contains all of the scripts necessary to recreate the phylogenetic augmentation analysis and associated figures.

## Download data
The data for the analyses are stored on Zenodo and can be downloaded from there.

This includes the following files:
* Drosophila S2 training, validation, and testing files
* Drosphila S2 homolog files
* Basset training, validation, and testing files
* Basset homolog files

Once downloaded, place these files in the `./input` directory.

## Initialize the conda environment
Loading from the conda file should create the same environment that was used to run the initial analysis. The analysis requires access to a GPU.

```
# Load the conda environment from file
conda env create -f environment.yml

# Activate conda environment
conda activate phylogenetic_augmentations

# Install R packages

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
# Train models on Drosophila S2 data
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

# Recreating some of the datasets

## Extracting homologs from hal alignments
```
# Foo
```

## Getting a sorted list of species from an alignment
```
# Foo
```