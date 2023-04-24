# Visualize the model performance for Drosophila S2 STARR-seq
library(tidyverse)
library(cowplot)

# Read correlation file
drosophila_corr_df <- read_tsv("./output_drosophila/model_correlation.tsv")
drosophila_corr_df$type <- factor(drosophila_corr_df$type)
drosophila_corr_df$type <- fct_relevel(drosophila_corr_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
drosophila_corr_df$type <- fct_recode(drosophila_corr_df$type, `No augmentations` = "none", `Augmentations + FT` = "homologs_finetune", `FT` = "finetune", `Augmentations` = "homologs")

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

# Summarize the data - VALIDATION
drosophila_corr_summary_dev_df <- data_summary(drosophila_corr_df, varname="pcc_val_Dev", 
                    groupnames=c("model", "type"))
plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=model, y=pcc_val_Dev, colour=type, fill=type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_val_Dev-sd, ymax = pcc_val_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('#000000','#E69F00', '#7fc97f', '#7393B3')) +
  xlab("Type") +
  ylab("Dev validation performance (PCC)")


drosophila_corr_summary_hk_df <- data_summary(drosophila_corr_df, varname="pcc_val_Hk", 
                                           groupnames=c("model", "type"))
plot_b <- ggplot(drosophila_corr_summary_hk_df, aes(x=model, y=pcc_val_Hk, colour=type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_val_Hk-sd, ymax = pcc_val_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('#000000','#E69F00', '#7fc97f', '#7393B3')) +
  xlab("Type") +
  ylab("Hk test performance (PCC)")

#plot_grid(plot_a, plot_b, labels = "AUTO")


# Summarize the data - TEST
drosophila_corr_summary_dev_df <- data_summary(drosophila_corr_df, varname="pcc_test_Dev", 
                                               groupnames=c("model", "type"))
plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=model, y=pcc_test_Dev, colour=type, fill=type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('#000000','#E69F00', '#7fc97f', '#7393B3')) +
  xlab("") +
  ylab("Test performance (PCC)") +
  ggtitle('Development enhancers') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14))


drosophila_corr_summary_hk_df <- data_summary(drosophila_corr_df, varname="pcc_test_Hk", 
                                              groupnames=c("model", "type"))
plot_b <- ggplot(drosophila_corr_summary_hk_df, aes(x=model, y=pcc_test_Hk, colour=type, fill=type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  theme_bw() +
  scale_color_manual(values=c('#000000','#E69F00', '#7fc97f', '#7393B3')) +
  xlab("") +
  ylab("") +
  ggtitle('Housekeeping enhancers') +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5)) +
  guides(colour=guide_legend(title="Type"), fill='none')

grobs <- ggplotGrob(plot_b)$grobs
plot_b <- plot_b + theme(legend.position="none")

# Combine plots
#plot <- plot_grid(plot_a, plot_b, labels = "AUTO")
plot <- plot_grid(plot_a, plot_b)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
final_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .1))
final_plot