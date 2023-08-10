# Visualize the model performance for Drosophila S2 STARR-seq
library(tidyverse)
library(cowplot)

# Read correlation file
drosophila_corr_df <- read_tsv("./output_drosophila_num_species_rev/model_correlation.tsv")
drosophila_corr_df$homolog_aug_type <- factor(drosophila_corr_df$homolog_aug_type)
drosophila_corr_df$homolog_aug_type <- fct_relevel(drosophila_corr_df$homolog_aug_type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
drosophila_corr_df$homolog_aug_type <- fct_recode(drosophila_corr_df$homolog_aug_type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune", `FT` = "finetune", `Phylo Aug` = "homologs")

drosophila_corr_df <- drosophila_corr_df %>% separate('model', c('model', 'num_species'), sep='_')
drosophila_corr_df$num_species <- as.integer(drosophila_corr_df$num_species)

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

# Summarize the data - TEST
drosophila_corr_summary_dev_df <- data_summary(drosophila_corr_df, varname="pcc_test_Dev", 
                                               groupnames=c("num_species", "homolog_aug_type"))
plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=num_species, y=pcc_test_Dev, colour=homolog_aug_type, fill=homolog_aug_type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.6656, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3', '#7fc97f','#E69F00')) +
  scale_x_continuous(breaks=seq(1, 20, by=1)) +
  xlab("") +
  ylab("Test set performance (PCC)") +
  ylim(0.62, 0.72) +
  ggtitle('Developmental task') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14), axis.text = element_text(size=12), legend.title = element_text(size=13),
        legend.text = element_text(size=13), panel.border = element_rect(colour = "black", fill=NA, size=1))


drosophila_corr_summary_hk_df <- data_summary(drosophila_corr_df, varname="pcc_test_Hk", 
                                              groupnames=c("num_species", "homolog_aug_type"))
plot_b <- ggplot(drosophila_corr_summary_hk_df, aes(x=num_species, y=pcc_test_Hk, colour=homolog_aug_type, fill=homolog_aug_type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.7487, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3', '#7fc97f','#E69F00')) +
  scale_x_continuous(breaks=seq(1, 20, by=1)) +
  xlab("") +
  ylab("Test set performance (PCC)") +
  ylim(0.73, 0.8) +
  ggtitle('Housekeeping task') +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14), axis.text = element_text(size=12), legend.title = element_text(size=12),
        legend.text.align=1, legend.text = element_text(size=11), panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  guides(colour=guide_legend(title="Type"), fill='none')

grobs <- ggplotGrob(plot_b)$grobs
plot_b <- plot_b + theme(legend.position="none")

# Combine plots
#plot <- plot_grid(plot_a, plot_b, labels = "AUTO")
plot <- plot_grid(plot_a, plot_b, ncol=1)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
final_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .1))
final_plot
#ggsave(file="./output/model_performance_one_homolog_per.pdf", plot=final_plot, width=6.75, height=9, units=c("in"))
