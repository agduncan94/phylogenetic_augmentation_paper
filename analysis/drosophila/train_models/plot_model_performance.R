# Visualize the model performance for Drosophila S2 STARR-seq
library(tidyverse)
library(cowplot)

# Read correlation file
drosophila_corr_df <- read_tsv("./output_3/model_correlation.tsv")
drosophila_corr_df$type <- factor(drosophila_corr_df$type)
drosophila_corr_df$type <- fct_relevel(drosophila_corr_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))

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
drosophila_corr_summary_dev_df <- data_summary(drosophila_corr_df, varname="pcc_val_dev", 
                    groupnames=c("model", "type"))
plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=model, y=pcc_val_dev, colour=type, fill=type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_val_dev-sd, ymax = pcc_val_dev+sd), width=.4, position=position_dodge(.9)) +
  theme_bw() +
  scale_color_manual(values=c('#000000','#E69F00', '#7fc97f', '#7393B3')) +
  xlab("Type") +
  ylab("Dev validation performance (PCC)")


drosophila_corr_summary_hk_df <- data_summary(drosophila_corr_df, varname="pcc_val_hk", 
                                           groupnames=c("model", "type"))
plot_b <- ggplot(drosophila_corr_summary_hk_df, aes(x=model, y=pcc_val_hk, colour=type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_val_hk-sd, ymax = pcc_val_hk+sd), width=.4, position=position_dodge(.9)) +
  theme_bw() +
  scale_color_manual(values=c('#000000','#E69F00', '#7fc97f', '#7393B3')) +
  xlab("Type") +
  ylab("Hk validation performance (PCC)")

plot_grid(plot_a, plot_b, labels = "AUTO")
