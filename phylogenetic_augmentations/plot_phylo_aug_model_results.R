# ####################################################################################################################
# plot_phylo_aug_model_results.R
#
# Visualize the test performance of different model architectures on the Drosophila S2 enhancer data and Basset data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
library(tidyverse)
library(cowplot)

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
drosophila_corr_df <- read_tsv("./output_drosophila_augs_rerun/model_correlation.tsv")

# Clean up values for display
drosophila_corr_df$homolog_aug_type <- factor(drosophila_corr_df$homolog_aug_type)
drosophila_corr_df$homolog_aug_type <- fct_relevel(drosophila_corr_df$homolog_aug_type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
drosophila_corr_df$homolog_aug_type <- fct_recode(drosophila_corr_df$homolog_aug_type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune", `FT` = "finetune", `Phylo Aug` = "homologs")
drosophila_corr_df$model <- factor(drosophila_corr_df$model)
drosophila_corr_df$model <- fct_relevel(drosophila_corr_df$model, c('deepstarr', 'explainn', 'motif_deepstarr'))
drosophila_corr_df$model <- fct_recode(drosophila_corr_df$model, `DeepSTARR` = "deepstarr", `ExplaiNN` = "explainn", `Motif DeepSTARR` = "motif_deepstarr")

# Create plot for Development task
drosophila_corr_summary_dev_df <- data_summary(drosophila_corr_df, varname="pcc_test_Dev", 
                                               groupnames=c("model", "homolog_aug_type"))
plot_dev <- ggplot(drosophila_corr_summary_dev_df, aes(x=model, y=pcc_test_Dev, colour=homolog_aug_type, fill=homolog_aug_type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('#A9A9A9', '#E69F00', '#7fc97f','#7393B3')) +
  xlab("") +
  ylab("Test set performance (PCC)") +
  ggtitle('Developmental task') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14), axis.text = element_text(size=12), legend.title = element_text(size=13),
        legend.text = element_text(size=13), panel.border = element_rect(colour = "black", fill=NA, size=1))

drosophila_corr_summary_hk_df <- data_summary(drosophila_corr_df, varname="pcc_test_Hk", 
                                              groupnames=c("model", "homolog_aug_type"))
plot_hk <- ggplot(drosophila_corr_summary_hk_df, aes(x=model, y=pcc_test_Hk, colour=homolog_aug_type, fill=homolog_aug_type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#E69F00', '#7fc97f','#7393B3')) +
  xlab("") +
  ylab("Test set performance (PCC)") +
  ggtitle('Housekeeping task') +
  theme(legend.position="right", plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14), axis.text = element_text(size=12), legend.title = element_text(size=12),
        legend.text = element_text(size=11), panel.border = element_rect(colour = "black", fill=NA, size=1),
        legend.background = element_rect(size=0.5, linetype="solid", colour="black", fill="white")) +
  guides(colour=guide_legend(title="Type"), fill='none')

grobs <- ggplotGrob(plot_hk)$grobs
plot_hk <- plot_hk + theme(legend.position="none")

# Combine plots
plot_a <- plot_grid(plot_dev, plot_hk, ncol=2)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]

# Load Basset data
basset_df <- read_tsv("./output_basset/model_metrics.tsv")
basset_df$type <- factor(basset_df$type)
basset_df$fraction <- factor(basset_df$fraction)
basset_df$type <- fct_relevel(basset_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
basset_df$type <- fct_recode(basset_df$type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune", `FT` = "finetune", `Phylo Aug` = "homologs")
basset_df$model <- fct_recode(basset_df$model, `Basset` = "basset")
basset_df <- basset_df %>% filter(fraction == 1.0)

# Create a plot for the Basset data
basset_summary_df <- data_summary(basset_df, varname="mean_test_pr", 
                                  groupnames=c("type", "model"))

plot_basset <- ggplot(basset_summary_df, aes(x=model, y=mean_test_pr, colour=type, fill=type)) +
  geom_point(data=basset_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = mean_test_pr-sd, ymax = mean_test_pr+sd), width=.2, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#E69F00', '#7fc97f','#7393B3')) +
  xlab('') +
  ylab("Test set performance (Avg AUPRC)") +
  ggtitle('Basset') +
  theme(legend.position="none", plot.title = element_text(hjust = 0.5, size=15),
        axis.title=element_text(size=13), axis.text = element_text(size = 13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

plot_b <- plot_grid(plot_basset, legend, ncol=2)
figure <- plot_grid(plot_a, plot_b, ncol=1, labels=c('A', 'B'), rel_heights = c(1, 1,  .1))

# Save a high quality and low quality image
ggsave("../figures/phylo_aug_figure_2.tiff", figure, units="in", width=4, height=4, device='tiff', dpi=350)
ggsave("../figures/phylo_aug_figure_2.jpg", figure, units="in", width=4, height=4)
