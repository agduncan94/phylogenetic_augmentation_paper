# Visualize the model performance for CHEF
library(tidyverse)
library(cowplot)

# Read correlation file
chef_corr_df <- read_tsv("./output_chef_one_homolog_6_species/model_correlation.tsv")
chef_corr_df$type <- factor(chef_corr_df$type)
chef_corr_df$type <- fct_relevel(chef_corr_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
chef_corr_df$type <- fct_recode(chef_corr_df$type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune", `FT` = "finetune", `Phylo Aug` = "homologs")

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
chef_corr_summary_h3k27ac_df <- data_summary(chef_corr_df, varname="pcc_test_h3k27ac_log2_enrichment", 
                                               groupnames=c("model", "type"))
plot_a <- ggplot(chef_corr_summary_h3k27ac_df, aes(x=model, y=pcc_test_h3k27ac_log2_enrichment, colour=type, fill=type)) +
  geom_point(data=chef_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_h3k27ac_log2_enrichment-sd, ymax = pcc_test_h3k27ac_log2_enrichment+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3', '#7fc97f','#E69F00')) +
  xlab("") +
  ylab("Test set performance (PCC)") +
  ggtitle('H3K27ac task') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14), axis.text = element_text(size=12), legend.title = element_text(size=13),
        legend.text = element_text(size=13), panel.border = element_rect(colour = "black", fill=NA, size=1))


chef_corr_summary_tf_sum_df <- data_summary(chef_corr_df, varname="pcc_test_tf_sum", 
                                              groupnames=c("model", "type"))
plot_b <- ggplot(chef_corr_summary_tf_sum_df, aes(x=model, y=pcc_test_tf_sum, colour=type, fill=type)) +
  geom_point(data=chef_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_tf_sum-sd, ymax = pcc_test_tf_sum+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3', '#7fc97f','#E69F00')) +
  xlab("") +
  ylab("Test set performance (PCC)") +
  ggtitle('TF Sum task') +
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
#ggsave(file="./output/chef_test_performance_6_species.pdf", plot=final_plot, width=6.75, height=9, units=c("in"))
