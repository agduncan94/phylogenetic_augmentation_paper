# This script will plot the results of tomtom

# FROM PAPER:
# Hit rate is calculated by measuring how many filters matched to at least one JASPAR motif.
# Average q-value is calculated by taking the average of the smallest q-values for each filter among its matches.

library(tidyverse)

# Load the tomtom data
tomtom_res <- read_tsv("~/Downloads/tomtom.tsv")

# Filter for significant hits
tomtom_res_filtered <- tomtom_res %>% filter(`E-value` < 0.05)
num_significant_hits <- length(unique(tomtom_res_filtered$Query_ID))

# Average q-value
tomtom_res_filtered %>% group_by(Query_ID) %>% summarise(avg_q_value = mean(`q-value`)) %>% ungroup()
