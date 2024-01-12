# ####################################################################################################################
# plot_yeast_3utr_motif_analysis.R
#
# Visualize the global importance of the PUF3 consensus motif on model predictions
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
library(tidyverse)
library(cowplot)

# ====================================================================================================================
# Main code
# ====================================================================================================================

# Load yeast motif data for augmented model
puf3_binding_preds_aug <- read_tsv("../output/puf3_predicted_binding_augmented.tsv")

# Clean up values for display
puf3_binding_preds_aug$Type <- factor(puf3_binding_preds_aug$Type)

# Plot figure a
figure_a <- ggplot(puf3_binding_preds_aug, aes(x=Type, y=Y_pred)) +
  stat_boxplot(geom="errorbar") +
  geom_boxplot(aes(fill=Type)) +
  theme_bw() +
  xlab("Type") +
  ylab("Predicted PUF3 Binding") +
  ggtitle('DeepSTARR \n Phylogenetic augmentation + Fine-tuning') +
  ylim(0,1) +
  theme(legend.position="none", plot.title = element_text(hjust = 0.5, size=10), axis.title=element_text(size=11),
        axis.text = element_text(size =10), legend.title = element_text(size=11), legend.text = element_text(size=11),
        panel.border = element_rect(colour = "black", fill=NA, size=1), legend.background = element_rect(size=0.5, linetype="solid", colour="black", fill="white"))


# Load yeast motif data for baseline model
puf3_binding_preds_baseline <- read_tsv("../output/puf3_predicted_binding_baseline.tsv")

# Clean up values for display
puf3_binding_preds_baseline$Type <- factor(puf3_binding_preds_baseline$Type)

# Plot figure b
figure_b <- ggplot(puf3_binding_preds_baseline, aes(x=Type, y=Y_pred)) +
  stat_boxplot(geom="errorbar") +
  geom_boxplot(aes(fill=Type)) +
  theme_bw() +
  xlab("Type") +
  ylab("Predicted PUF3 Binding") +
  ggtitle('DeepSTARR \n Baseline') +
  ylim(0,1) +
  theme(plot.title = element_text(hjust = 0.5, size=10), axis.title=element_text(size=11),
        axis.text = element_text(size =10), legend.title = element_text(size=11), legend.text = element_text(size=11),
        panel.border = element_rect(colour = "black", fill=NA, size=1), legend.direction="horizontal",
        legend.background = element_rect(size=0.5, linetype="solid", colour="black", fill="white"))

# Copy legend from figure b
grobs <- ggplotGrob(figure_b)$grobs
figure_b <- figure_b + theme(legend.position="none")
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]

# Create final figure
figure <- plot_grid(figure_a, figure_b, ncol=2, labels=c('', ''))
figure <- plot_grid(figure, legend, ncol=1, rel_heights = c(1, .1))

# Save figure
ggsave("../figures/suppl_figure_1.tiff", figure, units="in", width=7.5, height=5, device='tiff', dpi=350)
ggsave("../figures/suppl_figure_1.jpg", figure, units="in", width=7.5, height=5)

