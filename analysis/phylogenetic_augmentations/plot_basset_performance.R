library(tidyverse)
library(cowplot)

# Read the metric files
basset_df <- read_tsv("./output_basset_split_homologs/model_metrics.tsv")
basset_df$type <- factor(basset_df$type)
basset_df$fraction <- factor(basset_df$fraction)
basset_df$type <- fct_relevel(basset_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
basset_df <- basset_df %>% filter(type %in% c('none', 'homologs_finetune'))
basset_df$type <- fct_recode(basset_df$type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune", `FT` = "finetune", `Phylo Aug` = "homologs")

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
basset_summary_df <- data_summary(basset_df, varname="mean_test_pr", 
                                               groupnames=c("type", "fraction"))

ggplot(basset_summary_df, aes(x=fraction, y=mean_test_pr, colour=type, fill=type)) +
  geom_point(data=basset_df, size=2, position = position_dodge(width=0.2)) +
  geom_hline(yintercept=0.5317102, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#E69F00')) +
  xlab("Fraction of original training data") +
  #xlab("") +
  ylab("Test set performance (AUPRC)") +
  ggtitle('Basset') +
  theme(plot.title = element_text(hjust = 0.5, size=15),
        axis.title=element_text(size=13), axis.text = element_text(size = 13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1))

# Try distributions
basset_perf_df <- basset_df %>% select(1:5, 176:339)%>% gather("class", "measure", 6:169)
ggplot(basset_perf_df, aes(x=fraction, y=measure, colour=type, fill=type)) +
  geom_jitter(data=basset_perf_df, size=2, position = position_dodge(width=0.2)) +
  geom_hline(yintercept=0.5317102, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#E69F00')) +
  xlab("Fraction of original training data") +
  #xlab("") +
  ylab("Test set performance (AUPRC)") +
  ggtitle('Basset') +
  theme(plot.title = element_text(hjust = 0.5, size=15),
        axis.title=element_text(size=13), axis.text = element_text(size = 13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1))
