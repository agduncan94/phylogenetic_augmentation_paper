# Visualize the test performance of on the Drosophila S2 enhancer data using different hyperparameter values

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


# Investigate number of species
# Read correlation file
drosophila_corr_df <- read_tsv("./drosophila/output_drosophila_num_species/model_correlation.tsv")
drosophila_corr_df$type <- factor(drosophila_corr_df$type)
drosophila_corr_df$type <- fct_relevel(drosophila_corr_df$type, c('none', 'finetune', 'homologs', 'homologs_finetune'))
drosophila_corr_df$type <- fct_recode(drosophila_corr_df$type, `None` = "none", `Phylo Aug + FT` = "homologs_finetune", `FT` = "finetune", `Phylo Aug` = "homologs")

drosophila_corr_df <- drosophila_corr_df %>% separate('model', c('model', 'num_species'), sep='_')
drosophila_corr_df$num_species <- as.integer(drosophila_corr_df$num_species)


# Summarize the data - TEST
drosophila_corr_summary_dev_df <- data_summary(drosophila_corr_df, varname="pcc_test_Dev", 
                                               groupnames=c("num_species", "type"))
plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=num_species, y=pcc_test_Dev, colour=type, fill=type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.6656, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3')) +
  scale_x_continuous(breaks=seq(1, 20, by=1)) +
  xlab("Number of species") +
  ylab("Test set performance (PCC)") +
  #ylim(0.62, 0.72) +
  ggtitle('Developmental task') +
  theme(legend.position="none",
        plot.title = element_text(hjust = 0.5),
        axis.title=element_text(size=14), axis.text = element_text(size=12), legend.title = element_text(size=13),
        legend.text = element_text(size=13), panel.border = element_rect(colour = "black", fill=NA, size=1))


drosophila_corr_summary_hk_df <- data_summary(drosophila_corr_df, varname="pcc_test_Hk", 
                                              groupnames=c("num_species", "type"))
plot_b <- ggplot(drosophila_corr_summary_hk_df, aes(x=num_species, y=pcc_test_Hk, colour=type, fill=type)) +
  geom_point(data=drosophila_corr_df, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Hk-sd, ymax = pcc_test_Hk+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.7487, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3')) +
  scale_x_continuous(breaks=seq(1, 20, by=1)) +
  xlab("Number of species") +
  ylab("Test set performance (PCC)") +
  #ylim(0.73, 0.8) +
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
num_species_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .1))

# Plot homolog rate
drosophila_pcc <- read_tsv("./output_drosophila_homolog_rate/model_correlation.tsv")
drosophila_pcc$homolog_aug_type <- factor(drosophila_pcc$homolog_aug_type)
drosophila_pcc$homolog_rate <- factor(drosophila_pcc$homolog_rate)
drosophila_pcc$fraction <- factor(drosophila_pcc$fraction)
drosophila_pcc$homolog_aug_type <- fct_relevel(drosophila_pcc$homolog_aug_type, c('none', 'finetune', 'homologs', 'homologs_finetune'))

# Filter out homolog_aug_types not needed
drosophila_pcc <- drosophila_pcc %>% filter(homolog_aug_type %in% c('homologs_finetune', 'homologs'))
drosophila_pcc$homolog_aug_type <- fct_recode(drosophila_pcc$homolog_aug_type, `None` = "none", `FT` = "finetune", `Phylo Aug` = "homologs", `Phylo Aug + FT` = "homologs_finetune")


# Summarize developmental data and plot
drosophila_corr_summary_dev_df <- data_summary(drosophila_pcc, varname="pcc_test_Dev", 
                                               groupnames=c("homolog_aug_type", "homolog_rate"))

plot_a <- ggplot(drosophila_corr_summary_dev_df, aes(x=homolog_rate, y=pcc_test_Dev, colour=homolog_aug_type, fill=homolog_aug_type)) +
  geom_point(data=drosophila_pcc, size=2, position = position_dodge(width=0.9)) +
  geom_errorbar(aes(ymin = pcc_test_Dev-sd, ymax = pcc_test_Dev+sd), width=.4, position=position_dodge(.9), colour="black") +
  geom_hline(yintercept=0.6656, linetype="dashed", color = "red") +
  theme_bw() +
  scale_color_manual(values=c('darkgrey', '#7393B3')) +
  xlab("Phylo aug rate") +
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
  scale_color_manual(values=c('darkgrey', '#7393B3')) +
  xlab("Phylo aug rate") +
  #xlab("") +
  ylab("Test set performance (PCC)") +
  #ylab("") +
  ggtitle('Housekeeping task') +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5, size=15), axis.title=element_text(size=13),
        axis.text = element_text(size = 13), legend.title = element_text(size=13), legend.text = element_text(size=13),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  guides(colour=guide_legend(title="Type"), fill='none')

grobs <- ggplotGrob(plot_b)$grobs
plot_b <- plot_b + theme(legend.position="none")

# Create vertical plot
# Combine plots
plot <- plot_grid(plot_a, plot_b, ncol=2)

# Add shared legend
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]
aug_rate_plot <- plot_grid(plot, legend, nrow = 2, rel_heights = c(1, .1))

# Final plot
figure <- plot_grid(num_species_plot, aug_rate_plot, ncol=1, labels=c('A', 'B'), rel_heights = c(1, 0.7,  .1))
figure
