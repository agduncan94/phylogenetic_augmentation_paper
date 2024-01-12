# ####################################################################################################################
# plot_hyperparameter_analysis_suppl_num_species_rev.R
#
# Visualize the test performance of the Drosophila S2 enhancer data based on the number of species (reversed order)
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
library(tidyverse)
library(cowplot)
library(ggrepel)

# ====================================================================================================================
# Common functions
# ====================================================================================================================

# Summarize data (mean and standard deviation)
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}

# ====================================================================================================================
# Main code
# ====================================================================================================================

# Load Drosophila data
drosophila_corr_df <- read_tsv("../output/drosophila_num_species_rev_metrics.tsv")

# Clean up values for display
drosophila_corr_df$type <- factor(drosophila_corr_df$type)
drosophila_corr_df$type <- fct_relevel(drosophila_corr_df$type, c('homologs', 'homologs_finetune'))
drosophila_corr_df <- drosophila_corr_df %>% filter(type == 'homologs_finetune')
drosophila_corr_df$type <- fct_recode(drosophila_corr_df$type, `Phylogenetic Augmentation + Fine-tuning` = "homologs_finetune", `Phylogenetic Augmentation` = "homologs")

#drosophila_corr_df <- drosophila_corr_df %>% separate('model', c('model', 'species'), sep='_')
drosophila_corr_df$numspecies <- as.integer(drosophila_corr_df$species)

# Load species
species_df <- read_tsv("../input/all_drosophila_species_distances_desc.txt")
species_df <- tibble::rowid_to_column(species_df, "numspecies")
species_df <- species_df %>% mutate(numspecies=numspecies)

# Merge
drosophila_corr_df <- inner_join(drosophila_corr_df, species_df, by='numspecies')

# Filter species
drosophila_corr_df <- drosophila_corr_df %>% filter(numspecies %in% c(0,1,5,10,20,136))

# Create plot for Development task
drosophila_corr_summary_dev_df <- data_summary(drosophila_corr_df, varname="pcc_test_Dev", 
                                               groupnames=c("total_length", "numspecies", "type"))
plot_dev <- ggplot(drosophila_corr_summary_dev_df, aes(x=total_length, y=pcc_test_Dev, colour=type, fill=type)) +
  geom_point(data=drosophila_corr_summary_dev_df, size=2) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.661, linetype="dashed", color = "darkgrey") +
  geom_hline(yintercept=0.689, linetype="dashed", color = "red") +
  geom_text_repel(aes(label=numspecies), box.padding = 0.3, color="black") +
  theme_bw() +
  scale_color_manual(values=c('#7393B3')) +
  xlab("Total evolutionary distance \n (Substitutions per site)") +
  ylab("Test set performance (PCC)") +
  ggtitle('Developmental enhancer activity') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5, size=11),
        axis.title=element_text(size=11), axis.text = element_text(size= 11), legend.title = element_text(size= 11),
        legend.text = element_text(size= 11), panel.border = element_rect(colour = "black", fill=NA, size=1))

# Create plot for Housekeeping task
drosophila_corr_summary_hk_df <- data_summary(drosophila_corr_df, varname="pcc_test_Hk", 
                                              groupnames=c("total_length", "numspecies", "type"))
plot_hk <- ggplot(drosophila_corr_summary_hk_df, aes(x=total_length, y=pcc_test_Hk, colour=type, fill=type)) +
  geom_point(data=drosophila_corr_summary_hk_df, size=2) +
  geom_hline(yintercept=0.741, linetype="dashed", color = "darkgrey") +
  geom_hline(yintercept=0.778, linetype="dashed", color = "red") +
  geom_text_repel(aes(label=numspecies), box.padding = 0.2, color="black") +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, colour="black", position=position_dodge(.9)) +
  
  theme_bw() +
  scale_color_manual(values=c('#7393B3')) +
  xlab("Total evolutionary distance \n (Substitutions per site)") +
  ylab("Test set performance (PCC)") +
  ggtitle('Housekeeping enhancer activity') +
  theme(plot.title = element_text(hjust = 0.5, size=11),
        axis.title=element_text(size=11), axis.text = element_text(size= 11), legend.title = element_text(size= 11),
        legend.text.align=1, legend.text = element_text(size=11), panel.border = element_rect(colour = "black", fill=NA, size=1),
        legend.direction="horizontal",
        legend.background = element_rect(size=0.5, linetype="solid", colour="black", fill="white")) +
  guides(colour=guide_legend(title="Type"), fill='none')

# Copy legend
grobs <- ggplotGrob(plot_hk)$grobs
plot_hk <- plot_hk + theme(legend.position="none")
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]

# Create final figure
figure <- plot_grid(plot_dev, plot_hk, ncol=2)
figure <- plot_grid(figure, legend, ncol=1, rel_heights = c(1, .1))

# Plot figure
ggsave("../figures/suppl_figure_2.tiff", figure, units="in", width=7, height=4.5, device='tiff', dpi=350)
ggsave("../figures/suppl_figure_2.jpg", figure, units="in", width=7, height=4.5)

