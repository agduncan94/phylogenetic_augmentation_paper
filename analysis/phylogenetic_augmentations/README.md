# Phylogenetic augmentation analysis

This folder contains all of the scripts necessary to recreate the phylogenetic augmentation analysis and associated figures.

## Drosophila analysis
To run the Drosophila S2 analysis using the different CNN architectures, run the following commands:

```
# Train models
bash perform_drosophila_augmentation_analysis.sh

# Plot data
Rscript --vanilla plot_drosophila_performance.R
```

To run the sampling analysis, run the following commands:

```
# Train models
bash perform_drosophila_sampling_analysis.sh

# Plot data
Rscript --vanilla plot_drosophila_sampling_analysis.R
```

## CHEF analysis
To run the CHEF analysis using the DeepSTARR CNN, run the following commands:

```
# Train models
bash perform_chef_augmentation_analysis.sh

# Plot data
Rscript --vanilla plot_chef_performance.R
```