# This script will plot the results of tomtom

# FROM PAPER:
# Hit rate is calculated by measuring how many filters matched to at least one JASPAR motif.
# Average q-value is calculated by taking the average of the smallest q-values for each filter among its matches.

library(tidyverse)

# Load the tomtom data
tomtom_res <- read_tsv("~/Downloads/tomtom.tsv")

tomtom_res %>% filter('p-value' < 0.05)
