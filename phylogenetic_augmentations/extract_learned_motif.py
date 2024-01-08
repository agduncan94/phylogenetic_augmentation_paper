# ####################################################################################################################
# identifyImportantMotifs.py
#
# Given a set of conensus sequences for motifs, determines which are used by the CNN to make predictions
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import matplotlib.colors as mcolors
from keras.models import model_from_json
from Bio import SeqIO
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from random import randrange, randint
import seaborn as sns
import os
from scipy import stats

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
output_folder = "./output_3_utr_motif/"
motif_db_path = "./puf3_motif.txt"
background_sequence_path = output_folder + "fastas/background.sequence.fa"
fasta_path = output_folder + "fastas/"
image_path = output_folder + "figures/"
model_path_prefix = "../output/yeast_augmentation_deepstarr/DeepSTARR_rep1_frac1.0/Model_DeepSTARR_rep1_frac1.0_none"
num_sequences_per_motif = 1000
alphabet = "ACGT"

# Create folders
os.makedirs(output_folder, exist_ok=True)
os.makedirs(fasta_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)

# Set GPU id
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# ====================================================================================================================
# Helper functions
# ====================================================================================================================

def one_hot_encode(seqs, standardize=None):
    """One-hot encodes all the sequences"""
    one_hot_data = []
    for seq in seqs:
        seq_length = len(seq)
        if (standardize is not None):
            seq_length = int(standardize)
        one_hot_seq = np.zeros((seq_length, len(alphabet)))
        seq_length = min(seq_length, len(seq))

        for b in range(0, len(alphabet)):
            index = [j for j in range(0, seq_length) if seq[j] == alphabet[b]]
            one_hot_seq[index, b] = 1
        one_hot_data.append(one_hot_seq)
    one_hot_data = np.array(one_hot_data)
    return one_hot_data


def predict_activity(seqs):
    """Predict the values of the given sequences"""
    seqs_oh = one_hot_encode(seqs)
    X = np.nan_to_num(seqs_oh)
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    Y_pred = keras_model.predict(X_reshaped)
    return Y_pred


def create_background_sequences():
    """Create background sequences to insert motif consensus into and return predicted sequences"""
    seq_len = 200
    letters = ["A", "T", "C", "G"]

    fasta_lines = []
    sequences = []

    for x in range(1, num_sequences_per_motif + 1):
        seq_name = ">backbone_" + str(x)
        seq = ""
        for pos in range(1, seq_len+1):
            random_pos = randrange(4)
            seq += letters[random_pos]
        fasta_lines.append(seq_name)
        fasta_lines.append(seq)
        sequences.append(seq)

    # Write to file
    with open(background_sequence_path, 'w') as f:
        for item in fasta_lines:
            f.write("%s\n" % item)

    # Predict with model
    return predict_activity(sequences)


def create_sequences_with_motif(motif_id, consensus):
    """Create sequences with the motif consensus and scrambled motif inserted"""
    motif_sequences = []
    scrambled_motif_sequences = []

    # Insert into each background sequence and store to file
    fasta_sequences = SeqIO.parse(open(background_sequence_path), 'fasta')
    for fasta in fasta_sequences:
        sequence = str(fasta.seq)

        # Choose a random position
        position = randint(0, len(sequence) - len(consensus))

        # Insert motif into sequence at random position
        altered_sequence = insert_tfbs_into_sequence_at_pos(
            sequence, consensus, position)
        motif_sequences.append(altered_sequence)

        # Insert scrambled motif into sequence at random position
        scrambled_consensus = scramble_motif(consensus)
        altered_sequence = insert_tfbs_into_sequence_at_pos(
            sequence, scrambled_consensus, position)
        scrambled_motif_sequences.append(altered_sequence)

    # Predict with model
    return predict_activity(motif_sequences), predict_activity(scrambled_motif_sequences)


def plot_motif_activity(motif_id, motif_name, consensus, Y_pred_motifs, Y_pred_scrambled, Y_pred_background):
    """Plot the predicted activity for the three sets of sequences"""
    combined_pred_col = np.array(
        [Y_pred_motifs.flatten(), Y_pred_scrambled.flatten(), Y_pred_background.flatten()]).flatten()
    combined_type_col = np.array([[motif_name.upper()] * num_sequences_per_motif, ['Control']
                                 * num_sequences_per_motif, ['Background'] * num_sequences_per_motif]).flatten()
    activity_df = pd.DataFrame(
        {'Y_pred': combined_pred_col, 'Type': combined_type_col})

    my_pal = {motif_name.upper(): "#b2df8a",
              "Control": "#1f78b4", "Background": "#a6cee3"}
    ax = sns.boxplot(x='Type', y='Y_pred', data=activity_df, palette=my_pal)
    ax.set_title(motif_name)
    ax.set_xlabel('Type')
    ax.set_ylabel('Predicted ' + motif_name.upper() + ' binding')
    ax.set_ylim([0, 1])
    plt.savefig(image_path + '/' + motif_id + "_" +
                motif_name.replace('::', '-') + ".png", format='png')
    plt.clf()

    activity_df.to_csv(output_folder + '/' +
              motif_name.upper() + '_predicted_binding_baseline.tsv', sep='\t', index=False)

def insert_tfbs_into_sequence_at_pos(seq, consensus, pos):
    """Inserts a consensus sequence into a sequence at a given position"""
    return seq[0:pos] + consensus + seq[pos + len(consensus):len(seq)]

# Function to scramble a motif


def scramble_motif(seq):
    """Scrambles the given motif consensus sequence"""
    shuffled_motif_list = list(seq)
    random.shuffle(shuffled_motif_list)
    return ''.join(shuffled_motif_list)

# Function to compute Cohen's D


def cohen_d(x, y):
    """Computes the Cohen's D between two lists"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

# ====================================================================================================================
# Main code
# ====================================================================================================================


# Load keras model
keras_model = model_from_json(open(model_path_prefix + '.json').read())
keras_model.load_weights(model_path_prefix + '.h5')

# Create background sequences to insert motifs into
pred_background_activity = create_background_sequences()
print(stats.shapiro(pred_background_activity).pvalue)

# Load the motif database for JASPAR CORE 2022 Vertebrates
motif_df = pd.read_csv(motif_db_path, sep='\t')

# Columns for Cohen's D file
name_col = []
id_col = []
motif_vs_background_col = []
scrambled_motif_vs_background_col = []
motif_vs_background_pval_col = []
scrambled_motif_vs_background_pval_col = []
motif_vs_background_adj_pval_col = []
scrambled_motif_vs_background_adj_pval_col = []

# Perform importance analysis on each motif from the database
for index, row in motif_df.iterrows():
    motif_id = row['motif_id']
    motif_name = row['name']
    consensus = row['consensus'].upper()
    print(motif_id)

    name_col.append(motif_name)
    id_col.append(motif_id)

    # Create sequences by inserting the consensus motif (and scrambled) into the background sequences
    pred_sequence_activity, pred_sequence_scrambled_activity = create_sequences_with_motif(
        motif_id, consensus)

    # Plot the predicted activity of the motif vs scrambled vs background
    plot_motif_activity(motif_id, motif_name, consensus, pred_sequence_activity,
                        pred_sequence_scrambled_activity, pred_background_activity)

    # Measure effect size
    motif_vs_background_cd = cohen_d(
        pred_sequence_activity, pred_background_activity)
    scrambled_motif_vs_background_cd = cohen_d(
        pred_sequence_scrambled_activity, pred_background_activity)

    motif_vs_background_col.append(motif_vs_background_cd)
    scrambled_motif_vs_background_col.append(scrambled_motif_vs_background_cd)

    # Perform significance test
    test_type = 'less'
    if motif_vs_background_cd > 0:
        test_type = 'greater'
    else:
        test_type = 'less'
    #test_stat, pval = stats.ttest_ind(np.array(pred_sequence_activity).flatten(), np.array(pred_background_activity).flatten(), alternative=test_type)
    test_stat, pval = stats.wilcoxon(np.array(pred_sequence_activity).flatten(
    ), np.array(pred_background_activity).flatten(), alternative=test_type)
    motif_vs_background_pval_col.append(pval)
    motif_vs_background_adj_pval_col.append(pval * len(motif_df))

    test_type = 'less'
    if scrambled_motif_vs_background_cd > 0:
        test_type = 'greater'
    else:
        test_type = 'less'

    #test_stat, pval = stats.ttest_ind(np.array(pred_sequence_scrambled_activity).flatten(), np.array(pred_background_activity).flatten(), alternative=test_type)
    test_stat, pval = stats.wilcoxon(np.array(pred_sequence_scrambled_activity).flatten(
    ), np.array(pred_background_activity).flatten(), alternative=test_type)
    scrambled_motif_vs_background_pval_col.append(pval)
    scrambled_motif_vs_background_adj_pval_col.append(pval * len(motif_df))

# Save effect size results to file
motifs_cohen_d = pd.DataFrame(list(zip(id_col, name_col, motif_vs_background_col, scrambled_motif_vs_background_col, motif_vs_background_pval_col, motif_vs_background_adj_pval_col, scrambled_motif_vs_background_pval_col,
                              scrambled_motif_vs_background_adj_pval_col)), columns=['motif_id', 'name', 'motif_cd', 'control_cd', 'p_val_motif_vs_bg', 'adj_p_val_motif_vs_bg', 'p_val_control_vs_bg', 'adj_p_val_control_vs_bg'])
motif_name_df = pd.read_csv(motif_db_path, sep='\t')
result = pd.merge(motif_name_df, motifs_cohen_d,
                  how="inner", on=["motif_id", "motif_id"])
result.to_csv(output_folder + '/' +
              'tfbs.effect.size.tsv', sep='\t', index=False)