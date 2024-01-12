# ####################################################################################################################
# plot_phylo_aug_sampling_analysis.R
#
# Visualize the test performance of on the Drosophila S2 enhancer data and Basset data with sampling, along with
# the 3'UTR PUF3 binding test performance
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
drosophila_pcc <- read_tsv("../output/drosophila_sampling_metrics.tsv")

# Clean up values for display
drosophila_pcc$type <- factor(drosophila_pcc$type)
drosophila_pcc$fraction <- factor(drosophila_pcc$fraction)
drosophila_pcc$type <- fct_relevel(drosophila_pcc$type, c('none', 'homologs', 'homologs_finetune', 'finetune'))
drosophila_pcc <- drosophila_pcc %>% filter(type %in% c('none', 'homologs_finetune'))
drosophila_pcc$type <- fct_recode(drosophila_pcc$type, `Baseline` = "none", `Phylogenetic Augmentation + Fine-tuning` = "homologs_finetune")

# Create plot for Development task
drosophila_corr_summary_dev_df <- data_summary(drosophila_pcc, varname="pcc_test_Dev", 
                                               groupnames=c("type", "fraction"))

plot_dev <- ggplot(drosophila_corr_summary_dev_df, aes(x=fraction, y=pcc_test_Dev, colour=type, fill=type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.661, linetype="dashed", color = "darkgrey") +
  geom_hline(yintercept=0.689, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3')) +
  xlab("Fraction of original training data") +
  ylab("Test set performance (PCC)") +
  ggtitle('Developmental enhancer activity') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5, size=11),
        axis.title=element_text(size=11), axis.text = element_text(size =10), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

# Create plot for Housekeeping task
drosophila_corr_summary_hk_df <- data_summary(drosophila_pcc, varname="pcc_test_Hk", 
                                              groupnames=c("type", "fraction"))
plot_hk <- ggplot(drosophila_corr_summary_hk_df, aes(x=fraction, y=pcc_test_Hk, colour=type, fill=type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.741, linetype="dashed", color = "darkgrey") +
  geom_hline(yintercept=0.778, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3')) +
  xlab("Fraction of original training data") +
  ylab("Test set performance (PCC)") +
  ggtitle('Housekeeping enhancer activity') +
  theme(legend.position="none", plot.title = element_text(hjust = 0.5, size=11), axis.title=element_text(size=11),
        axis.text = element_text(size =10), legend.title = element_text(size=11), legend.text = element_text(size=11),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

# Combine plots
plot_a <- plot_grid(plot_dev, plot_hk, ncol=2, labels=c('A', ''))

# Load Basset data
basset_df <- read_tsv("../output/basset_sampling_metrics.tsv")

# Clean up values for display
basset_df$type <- factor(basset_df$type)
basset_df$fraction <- factor(basset_df$fraction)
basset_df$type <- fct_relevel(basset_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
basset_df <- basset_df %>% filter(type %in% c('none', 'homologs_finetune'))
basset_df$type <- fct_recode(basset_df$type, `Baseline` = "none", `Phylogenetic Augmentation + Fine-tuning` = "homologs_finetune", `Fine-tuning` = "finetune", `Phylogenetic Augmentation` = "homologs")
basset_df$model <- fct_recode(basset_df$model, `Basset` = "basset")

# Create a plot for the Basset data
basset_summary_df <- data_summary(basset_df, varname="mean_test_pr", 
                                  groupnames=c("type", "fraction"))

plot_basset <- ggplot(basset_summary_df, aes(x=fraction, y=mean_test_pr, colour=type, fill=type)) +
  geom_point(data=basset_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = mean_test_pr-sd, ymax = mean_test_pr+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  geom_hline(yintercept=0.536, linetype="dashed", color = "darkgrey") +
  geom_hline(yintercept=0.575, linetype="dashed", color = "red") +
  scale_color_manual(values=c('darkgrey','#7393B3')) +
  xlab("Fraction of original training data") +
  ylab("Test set performance (Avg AUPRC)") +
  ggtitle('Basset') +
  theme(legend.position="none", plot.title = element_text(hjust = 0.5, size=11),
        axis.title=element_text(size=11), axis.text = element_text(size =10), legend.text = element_text(size=11),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

# Load yeast data
yeast_corr_df <- read_tsv("../output/yeast_model_metrics.tsv")

# Clean up values for display
yeast_corr_df$type <- factor(yeast_corr_df$type)
yeast_corr_df$type <- fct_relevel(yeast_corr_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
yeast_corr_df <- yeast_corr_df %>% filter(type %in% c('none', 'homologs_finetune'))
yeast_corr_df$type <- fct_recode(yeast_corr_df$type, `Baseline` = "none", `Phylogenetic Augmentation + Fine-tuning` = "homologs_finetune", `Fine-tuning` = "finetune", `Phylogenetic Augmentation` = "homologs")
yeast_corr_df$model <- factor(yeast_corr_df$model)
yeast_corr_df$model <- fct_relevel(yeast_corr_df$model, c('deepstarr', 'explainn', 'motif_deepstarr'))
yeast_corr_df$model <- fct_recode(yeast_corr_df$model, `DeepSTARR` = "deepstarr", `ExplaiNN` = "explainn", `Motif DeepSTARR` = "motif_deepstarr", `Scrambled Control` = "Scrambled Control")

# Create plot for yeast classification
yeast_corr_summary_df <- data_summary(yeast_corr_df, varname="pr_multilabel_test", 
                                      groupnames=c("model", "type"))
plot_yeast <- ggplot(yeast_corr_summary_df, aes(x=model, y=pr_multilabel_test, colour=type, fill=type)) +
  geom_point(data=yeast_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pr_multilabel_test-sd, ymax = pr_multilabel_test+sd), width=.1, position=position_dodge(.9), colour="black") +
  theme_bw() +
  geom_hline(yintercept=0.104, linetype="dashed", color = "darkgrey") +
  geom_hline(yintercept=0.448, linetype="dashed", color = "red") +
  scale_color_manual(values=c('#A9A9A9' ,'#7393B3')) +
  xlab("") +
  ylab("Test set performance (AUPRC)") +
  ggtitle('PUF3 binding') +
  theme(plot.title = element_text(hjust = 0.5, size=11),
        axis.title=element_text(size=11), axis.text = element_text(size=10), legend.title = element_text(size=11),
        legend.text = element_text(size=11), panel.border = element_rect(colour = "black", fill=NA, size=1), 
        legend.background = element_rect(size=0.5, linetype="solid", colour="black", fill="white"), legend.direction="horizontal") +
  guides(colour=guide_legend(title="Type"), fill='none')

# Copy legend
grobs <- ggplotGrob(plot_yeast)$grobs
plot_yeast <- plot_yeast + theme(legend.position="none")
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]

# Create final figure
plot_b_c <- plot_grid(plot_basset, plot_yeast, ncol=2, labels=c('B', 'C'))
figure <- plot_grid(plot_a, plot_b_c, ncol=1, rel_heights = c(1, 1,  .1))
figure <- plot_grid(figure, legend, ncol=1, rel_heights = c(1, .1))
figure <- figure + theme(panel.background = element_rect(fill = 'white', colour = 'white'))

# Save a high quality and low quality image
ggsave("../figures/figure_3.tiff", figure, units="in", width=7.5, height=7.5, device='tiff', dpi=350)
ggsave("../figures/figure_3.jpg", figure, units="in", width=7.5, height=7.5)
