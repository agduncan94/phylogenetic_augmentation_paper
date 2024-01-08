# ####################################################################################################################
# plot_phylo_aug_hyperparameter_analysis.R
#
# Visualize the test performance of on the Drosophila S2 enhancer data using different hyperparameter values
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
drosophila_corr_df <- read_tsv("../output/drosophila_num_species_new_metrics.tsv")

# Clean up values for display
drosophila_corr_df$type <- factor(drosophila_corr_df$type)
drosophila_corr_df$type <- fct_relevel(drosophila_corr_df$type, c('homologs', 'homologs_finetune'))
drosophila_corr_df <- drosophila_corr_df %>% filter(type == 'homologs_finetune')
drosophila_corr_df$type <- fct_recode(drosophila_corr_df$type, `Phylogenetic Augmentation + Finetuning` = "homologs_finetune", `Phylogenetic Augmentation` = "homologs")

drosophila_corr_df <- drosophila_corr_df %>% separate('model', c('model', 'species'), sep='_')
drosophila_corr_df$numspecies <- as.integer(drosophila_corr_df$species)

# Load species
species_df <- read_tsv("../input/all_drosphila_species_distances_asc.txt")
species_df <- tibble::rowid_to_column(species_df, "numspecies")
species_df <- species_df %>% mutate(numspecies=numspecies-1)

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
  #scale_x_continuous(limits = c(0, 2.5)) +
  #scale_y_continuous(limits = c(0.66, 0.705)) +
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
  #scale_x_continuous(limits = c(0, 2.5)) +
  #scale_y_continuous(limits = c(0.73, 0.785)) +
  xlab("Total evolutionary distance \n (Substitutions per site)") +
  ylab("Test set performance (PCC)") +
  ggtitle('Housekeeping enhancer activity') +
  theme(legend.position="none", plot.title = element_text(hjust = 0.5, size=11),
        axis.title=element_text(size=11), axis.text = element_text(size= 11), legend.title = element_text(size= 11),
        legend.text.align=1, legend.text = element_text(size=11), panel.border = element_rect(colour = "black", fill=NA, size=1))

plot_a <- plot_grid(plot_dev, plot_hk, ncol=2)

# Plot homolog rate
drosophila_pcc <- read_tsv("../output/drosophila_phylo_aug_rate_metrics.tsv")
drosophila_pcc$type <- factor(drosophila_pcc$type)
drosophila_pcc$homolog_rate <- factor(drosophila_pcc$homolog_rate)
drosophila_pcc$fraction <- factor(drosophila_pcc$fraction)
drosophila_pcc$type <- fct_relevel(drosophila_pcc$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))

# Filter out types not needed
drosophila_pcc <- drosophila_pcc %>% filter(type %in% c('homologs_finetune'))
drosophila_pcc$type <- fct_recode(drosophila_pcc$type, `Baseline` = "none", `Finetuning` = "finetune", `Phylogenetic Augmentation` = "homologs", `Phylogenetic Augmentation + Finetuning` = "homologs_finetune")


# Summarize developmental data and plot
drosophila_corr_summary_dev_df <- data_summary(drosophila_pcc, varname="pcc_test_Dev", 
                                               groupnames=c("type", "homolog_rate"))

plot_dev <- ggplot(drosophila_corr_summary_dev_df, aes(x=homolog_rate, y=pcc_test_Dev, colour=type, fill=type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.661, linetype="dashed", color = "darkgrey") +
  theme_bw() +
  geom_hline(yintercept=0.661, linetype="dashed", color = "darkgrey") +
  geom_hline(yintercept=0.689, linetype="dashed", color = "red") +
  scale_color_manual(values=c('#7393B3')) +
  xlab("Phylogenetic augmentation rate") +
  ylab("Test set performance (PCC)") +
  ggtitle('Developmental enhancer activity') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5, size=11),
        axis.title=element_text(size=11), axis.text = element_text(size = 11), legend.text = element_text(size= 11),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

# Summarize housekeeping data and plot
drosophila_corr_summary_hk_df <- data_summary(drosophila_pcc, varname="pcc_test_Hk", 
                                              groupnames=c("type", "homolog_rate"))
plot_hk <- ggplot(drosophila_corr_summary_hk_df, aes(x=homolog_rate, y=pcc_test_Hk, colour=type, fill=type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.741, linetype="dashed", color = "darkgrey") +
  theme_bw() +
  geom_hline(yintercept=0.741, linetype="dashed", color = "darkgrey") +
  geom_hline(yintercept=0.778, linetype="dashed", color = "red") +
  scale_color_manual(values=c('#7393B3')) +
  xlab("Phylogenetic augmentation rate") +
  ylab("Test set performance (PCC)") +
  ggtitle('Housekeeping enhancer activity') +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5, size=11), axis.title=element_text(size=11),
        axis.text = element_text(size = 11), legend.title = element_text(size= 11), legend.text = element_text(size= 11),
        panel.border = element_rect(colour = "black", fill=NA, size=1), legend.background = element_rect(size=0.5, linetype="solid", colour="black", fill="white"), legend.direction="horizontal") +
  guides(colour=guide_legend(title="Type"), fill='none')

grobs <- ggplotGrob(plot_hk)$grobs
plot_hk <- plot_hk + theme(legend.position="none")

# Combine plots
plot_b <- plot_grid(plot_dev, plot_hk, ncol=2)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]

# Final plot
figure <- plot_grid(plot_a, plot_b, ncol=1, labels=c('A', 'B'), rel_heights = c(1, 1))
figure <- plot_grid(figure, legend, ncol=1, rel_heights = c(1, .1))
figure

# Save a high quality and low quality image
ggsave("../figures/phylo_aug_figure_4_new.tiff", figure, units="in", width=6, height=6, device='tiff', dpi=350)
ggsave("../figures/phylo_aug_figure_4_new.jpg", figure, units="in", width=7.5, height=7)
