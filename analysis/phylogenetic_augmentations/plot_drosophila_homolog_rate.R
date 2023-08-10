# Visualize the performance of DeepSTARR with and without homologs

# Load libraries
library(tidyverse)
library(cowplot)

drosophila_pcc <- read_tsv("./output_drosophila_homolog_rate/model_correlation.tsv")
drosophila_pcc$homolog_aug_type <- factor(drosophila_pcc$homolog_aug_type)
drosophila_pcc$homolog_rate <- factor(drosophila_pcc$homolog_rate)
drosophila_pcc$fraction <- factor(drosophila_pcc$fraction)
drosophila_pcc$homolog_aug_type <- fct_relevel(drosophila_pcc$homolog_aug_type, c('none', 'finetune', 'homologs', 'homologs_finetune'))

# Filter out homolog_aug_types not needed
drosophila_pcc <- drosophila_pcc %>% filter(homolog_aug_type %in% c('homologs_finetune'))
drosophila_pcc$homolog_aug_type <- fct_recode(drosophila_pcc$homolog_aug_type, `None` = "none", `FT` = "finetune", `Phylo Aug` = "homologs", `Phylo Aug + FT` = "homologs_finetune")

# Function to summarize data
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

# Summarize developmental data and plot
drosophila_corr_summary_dev_df <- data_summary(drosophila_pcc, varname="pcc_test_Dev", 
                                               groupnames=c("homolog_aug_type", "homolog_rate"))

plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=homolog_rate, y=pcc_test_Dev, colour=homolog_aug_type, fill=homolog_aug_type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.6656, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#E69F00', '#7fc97f','#7393B3')) +
  xlab("Homolog rate") +
  #xlab("") +
  ylab("Test set performance (PCC)") +
  ggtitle('Developmental task') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5, size=15),
        axis.title=element_text(size=13), axis.text = element_text(size = 13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

# Summarize housekeeping data and plot
drosophila_corr_summary_hk_df <- data_summary(drosophila_pcc, varname="pcc_test_Hk", 
                                              groupnames=c("homolog_aug_type", "homolog_rate"))
plot_b <- ggplot(drosophila_corr_summary_hk_df, aes(x=homolog_rate, y=pcc_test_Hk, colour=homolog_aug_type, fill=homolog_aug_type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.7487, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#E69F00', '#7fc97f','#7393B3')) +
  xlab("Homolog rate") +
  #xlab("") +
  ylab("Test set performance (PCC)") +
  #ylab("") +
  ggtitle('Housekeeping task') +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5, size=15), axis.title=element_text(size=13),
        axis.text = element_text(size = 13), legend.title = element_text(size=13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  guides(colour=guide_legend(title="homolog_aug_type"), fill='none')

grobs <- ggplotGrob(plot_b)$grobs
plot_b <- plot_b + theme(legend.position="none")

# Create vertical plot
# Combine plots
plot <- plot_grid(plot_a, plot_b,ncol=1)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
final_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .1))
#final_plot

#final_plot <- ggdraw(add_sub(final_plot, "Fraction of original training data", vpadding=grid::unit(0,"lines"),y=9.5, x=0.5, vjust=4.5))
final_plot
#ggsave(file="./drosophila/output/sampled_training_data_test_performance_vertical.png", plot=final_plot, width=6.75, height=9)

# Create horizontal plot
# Combine plots
plot <- plot_grid(plot_a, plot_b)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
final_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .1))
#final_plot

#final_plot <- ggdraw(add_sub(final_plot, "Fraction of original training data", vpadding=grid::unit(0,"lines"),y=9.5, x=0.5, vjust=4.5))
final_plot
#ggsave(file="./drosophila/output/sampled_training_data_test_performance_horizontal.png", plot=final_plot, width=10, height=5.75)
