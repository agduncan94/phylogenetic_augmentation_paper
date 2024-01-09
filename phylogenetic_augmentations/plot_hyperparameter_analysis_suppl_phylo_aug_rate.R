# ####################################################################################################################
# plot_hyperparameter_analysis_suppl_phylo_aug_rate.R
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

# Plot homolog rate
drosophila_pcc <- read_tsv("../output/drosophila_phylo_aug_rate_reduced_species/model_metrics.tsv")
drosophila_pcc$type <- factor(drosophila_pcc$type)
drosophila_pcc$phylo_aug_rate <- factor(drosophila_pcc$phylo_aug_rate)
drosophila_pcc$fraction <- factor(drosophila_pcc$fraction)
drosophila_pcc$type <- fct_relevel(drosophila_pcc$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))

# Filter out types not needed
drosophila_pcc <- drosophila_pcc %>% filter(type %in% c('homologs_finetune'))
drosophila_pcc$type <- fct_recode(drosophila_pcc$type, `Baseline` = "none", `Finetuning` = "finetune", `Phylogenetic Augmentation` = "homologs", `Phylogenetic Augmentation + Finetuning` = "homologs_finetune")


# Summarize developmental data and plot
drosophila_corr_summary_dev_df <- data_summary(drosophila_pcc, varname="pcc_test_Dev", 
                                               groupnames=c("type", "phylo_aug_rate"))

plot_dev <- ggplot(drosophila_corr_summary_dev_df, aes(x=phylo_aug_rate, y=pcc_test_Dev, colour=type, fill=type)) +
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
                                              groupnames=c("type", "phylo_aug_rate"))
plot_hk <- ggplot(drosophila_corr_summary_hk_df, aes(x=phylo_aug_rate, y=pcc_test_Hk, colour=type, fill=type)) +
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
plot <- plot_grid(plot_dev, plot_hk, ncol=2)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]

# Final plot
figure <- plot_grid(plot, legend, ncol=1, rel_heights = c(1, .1))
figure
ggsave("../figures/phylo_aug_suppl_figure_3.jpg", figure, units="in", width=7.5, height=4)
