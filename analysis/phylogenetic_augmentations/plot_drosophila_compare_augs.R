# Visualize the model performance for Drosophila S2 STARR-seq
library(tidyverse)
library(cowplot)

# Read correlation file
drosophila_corr_df <- read_tsv("./output_drosophila_augs_compare/model_correlation.tsv")
drosophila_corr_df$homolog_aug_type <- factor(drosophila_corr_df$homolog_aug_type)
drosophila_corr_df$homolog_aug_type <- fct_relevel(drosophila_corr_df$homolog_aug_type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
drosophila_corr_df <- drosophila_corr_df %>% filter(homolog_aug_type %in% c('none', 'homologs_finetune'))
drosophila_corr_df$homolog_aug_type <- fct_recode(drosophila_corr_df$homolog_aug_type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune", `FT` = "finetune", `Phylo Aug` = "homologs")

drosophila_corr_df$aug_type <- factor(drosophila_corr_df$aug_type)
drosophila_corr_df$aug_type <- fct_recode(drosophila_corr_df$aug_type, `None` = "positive_peaks-negative", `Flanks` = "peak_849bp_region-positive_peaks-negative", `Other` = "Other-positive_peaks-negative", `Flanks + Other` = "peak_849bp_region-Other-positive_peaks-negative")


plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=aug_type, y=pcc_test_Dev, colour=homolog_aug_type, fill=aug_type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3', '#7fc97f','#E69F00')) +
  xlab("") +
  ylab("Test set performance (PCC)") +
  ggtitle('Developmental task') +
  #geom_hline(yintercept=0.6410585, linetype="dashed", color = "red") +
  #geom_hline(yintercept=0.6766377, linetype="dashed", color = "red") +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14), axis.text = element_text(size=12), legend.title = element_text(size=13),
        legend.text = element_text(size=13), panel.border = element_rect(colour = "black", fill=NA, size=1))


plot_b <- ggplot(drosophila_corr_summary_dev_df, aes(x=aug_type, y=pcc_test_Hk, colour=homolog_aug_type, fill=aug_type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3', '#7fc97f','#E69F00')) +
  xlab("") +
  ylab("Test set performance (PCC)") +
  ggtitle('Housekeeping task') +
  #geom_hline(yintercept=0.7805010, linetype="dashed", color = "red") +
  #geom_hline(yintercept=0.7945047, linetype="dashed", color = "red") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14), axis.text = element_text(size=12), legend.title = element_text(size=13),
        legend.text = element_text(size=13), panel.border = element_rect(colour = "black", fill=NA, size=1))+
  guides(colour=guide_legend(title="Type"), fill='none')

grobs <- ggplotGrob(plot_b)$grobs
plot_b <- plot_b + theme(legend.position="none")

# Combine plots
#plot <- plot_grid(plot_a, plot_b, labels = "AUTO")
plot <- plot_grid(plot_a, plot_b, ncol=1)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
final_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .2))
final_plot
#ggsave(file="./output/model_performance_one_homolog_per.pdf", plot=final_plot, width=6.75, height=9, units=c("in"))
