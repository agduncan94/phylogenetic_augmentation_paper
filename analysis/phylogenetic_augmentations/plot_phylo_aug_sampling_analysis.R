# Visualize the test performance of on the Drosophila S2 enhancer data and Basset data with sampling

# Import libraries
library(tidyverse)
library(cowplot)

# Common functions
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

# Load Drosophila data
drosophila_pcc <- read_tsv("./drosophila/output_drosophila_sampling/model_correlation.tsv")

# Clean up values for display
drosophila_pcc$type <- factor(drosophila_pcc$type)
drosophila_pcc$fraction <- factor(drosophila_pcc$fraction)
drosophila_pcc$type <- fct_relevel(drosophila_pcc$type, c('none', 'homologs', 'homologs_finetune', 'finetune'))
drosophila_pcc <- drosophila_pcc %>% filter(type %in% c('none', 'homologs_finetune'))
drosophila_pcc$type <- fct_recode(drosophila_pcc$type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune")


# Create plot for Development task
drosophila_corr_summary_dev_df <- data_summary(drosophila_pcc, varname="pcc_test_Dev", 
                                               groupnames=c("type", "fraction"))

plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=fraction, y=pcc_test_Dev, colour=type, fill=type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.6590672, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3')) +
  xlab("Fraction of original training data") +
  #xlab("") +
  ylab("Test set performance (PCC)") +
  ggtitle('Developmental task') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5, size=15),
        axis.title=element_text(size=13), axis.text = element_text(size = 13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

# Create plot for Housekeeping task
drosophila_corr_summary_hk_df <- data_summary(drosophila_pcc, varname="pcc_test_Hk", 
                                              groupnames=c("type", "fraction"))
plot_b <- ggplot(drosophila_corr_summary_hk_df, aes(x=fraction, y=pcc_test_Hk, colour=type, fill=type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.7440964, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3')) +
  xlab("Fraction of original training data") +
  #xlab("") +
  ylab("Test set performance (PCC)") +
  #ylab("") +
  ggtitle('Housekeeping task') +
  theme(legend.position="right", plot.title = element_text(hjust = 0.5, size=15), axis.title=element_text(size=13),
        axis.text = element_text(size = 13), legend.title = element_text(size=13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1), legend.background = element_rect(size=0.5, linetype="solid", colour="black", fill="white")) +
  guides(colour=guide_legend(title="Type"), fill='none')

grobs <- ggplotGrob(plot_b)$grobs
plot_b <- plot_b + theme(legend.position="none")

# Create vertical plot
# Combine plots
plot <- plot_grid(plot_a, plot_b, ncol=2)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
final_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .1))
#final_plot

drosophila_plot <- plot



# Load Basset data
basset_df <- read_tsv("./output_basset/model_metrics.tsv")
basset_df$type <- factor(basset_df$type)
basset_df$fraction <- factor(basset_df$fraction)
basset_df$type <- fct_relevel(basset_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
basset_df <- basset_df %>% filter(type %in% c('none', 'homologs_finetune'))

basset_df$type <- fct_recode(basset_df$type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune", `FT` = "finetune", `Phylo Aug` = "homologs")
basset_df$model <- fct_recode(basset_df$model, `Basset` = "basset")



# Create a plot for the Basset data
basset_summary_df <- data_summary(basset_df, varname="mean_test_pr", 
                                  groupnames=c("type", "fraction"))

plot_c <- ggplot(basset_summary_df, aes(x=fraction, y=mean_test_pr, colour=type, fill=type)) +
  geom_point(data=basset_df, size=2, position = position_dodge(width=0.2)) +
  geom_errorbar(aes(ymin = mean_test_pr-sd, ymax = mean_test_pr+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  geom_hline(yintercept=0.5359999, linetype="dashed", color = "red") +
  scale_color_manual(values=c('darkgrey','#7393B3')) +
  xlab("Fraction of original training data") +
  ylab("Test set performance (Avg AUPRC)") +
  ggtitle('Basset') +
  theme(legend.position="none", plot.title = element_text(hjust = 0.5, size=15),
        axis.title=element_text(size=13), axis.text = element_text(size = 13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

basset_plot <- plot_grid(plot_c, legend, ncol=2)
figure <- plot_grid(drosophila_plot, basset_plot, ncol=1, labels=c('A', 'B'), rel_heights = c(1, 1,  .1))
figure
