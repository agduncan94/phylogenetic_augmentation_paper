
# Visualize the model performance for CHEF
library(tidyverse)
library(cowplot)

# Read correlation file
chef_corr_df <- read_tsv("./output_chef/model_correlation.tsv")
chef_corr_df$type <- factor(chef_corr_df$type)
chef_corr_df$type <- fct_relevel(chef_corr_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))

# Filter out types not needed
chef_corr_df <- chef_corr_df %>% filter(type %in% c('none', 'homologs_finetune'))
chef_corr_df$type <- fct_recode(chef_corr_df$type, `No augmentations` = "none", `Augmentations` = "homologs_finetune")


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

# Summarize the data
chef_corr_summary_h3k27ac_df <- data_summary(chef_corr_df, varname="pcc_test_h3k27ac_log2_enrichment", 
                                               groupnames=c("model", "type"))
plot_a <- ggplot(chef_corr_summary_h3k27ac_df, aes(x=model, y=pcc_test_h3k27ac_log2_enrichment, colour=type, fill=type)) +
  geom_point(data=chef_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_h3k27ac_log2_enrichment-sd, ymax = pcc_test_h3k27ac_log2_enrichment+sd), width=.4, position=position_dodge(.9)) +
  theme_bw() +
  scale_color_manual(values=c('#000000','#E69F00', '#7fc97f', '#7393B3')) +
  xlab('') +
  ylab("h3k27ac test set performance (PCC)") +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5)) +
  guides(colour=guide_legend(title="Type"), fill='none') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14))
  #xlab("Type") +


chef_corr_summary_tf_sum_df <- data_summary(chef_corr_df, varname="pcc_test_tf_sum", 
                                              groupnames=c("model", "type"))
plot_b <- ggplot(chef_corr_summary_tf_sum_df, aes(x=model, y=pcc_test_tf_sum, colour=type)) +
  geom_point(data=chef_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_tf_sum-sd, ymax = pcc_test_tf_sum+sd), width=.4, position=position_dodge(.9)) +
  theme_bw() +
  scale_color_manual(values=c('#000000','#E69F00', '#7fc97f', '#7393B3')) +
  xlab("") +
  ylab("TF sum test set performance (PCC)") +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5), axis.title=element_text(size=14)) +
  guides(colour=guide_legend(title="Type"), fill='none')

grobs <- ggplotGrob(plot_b)$grobs
plot_b <- plot_b + theme(legend.position="none")

plot <- plot_grid(plot_a, plot_b)

legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
final_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .1))
#final_plot

final_plot <- ggdraw(add_sub(final_plot, "DeepSTARR", vpadding=grid::unit(0,"lines"),y=9, x=0.5, vjust=4.5))
final_plot
