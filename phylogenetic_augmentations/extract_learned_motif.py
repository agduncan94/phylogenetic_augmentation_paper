# ####################################################################################################################
# extract_learned_motif.py
#
# Creates three sets of sequences to test PUF3 importance and get model predictions
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
from keras.models import model_from_json
from Bio import SeqIO
import numpy as np
import pandas as pd
import random
from random import randrange, randint
import os

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
output_folder = "../output/puf3_motif_importance/"
motif_db_path = "../input/puf3_motif.txt"
background_sequence_path = output_folder + "fastas/background.sequence.fa"
fasta_path = output_folder + "fastas/"
model_prefixes = {
    'baseline': "../output/yeast_augmentation/DeepSTARR_rep1_frac1.0/Model_DeepSTARR_rep1_frac1.0_none",
    'augmented': "../output/yeast_augmentation/DeepSTARR_rep1_frac1.0/Model_DeepSTARR_rep1_frac1.0_homologs_finetune"
}
num_sequences_per_motif = 1000
alphabet = "ACGT"

# Create folders
os.makedirs(output_folder, exist_ok=True)
os.makedirs(fasta_path, exist_ok=True)

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


def create_file_for_plotting(motif_name, model_type, Y_pred_motifs, Y_pred_scrambled, Y_pred_background):
    """Plot the predicted activity for the three sets of sequences"""
    combined_pred_col = np.array(
        [Y_pred_motifs.flatten(), Y_pred_scrambled.flatten(), Y_pred_background.flatten()]).flatten()
    combined_type_col = np.array([[motif_name.upper()] * num_sequences_per_motif, ['Control']
                                 * num_sequences_per_motif, ['Background'] * num_sequences_per_motif]).flatten()
    activity_df = pd.DataFrame(
        {'Y_pred': combined_pred_col, 'Type': combined_type_col})

    activity_df.to_csv(output_folder + '/' +
              motif_name.upper() + '_predicted_binding_' + model_type + '.tsv', sep='\t', index=False)

def insert_tfbs_into_sequence_at_pos(seq, consensus, pos):
    """Inserts a consensus sequence into a sequence at a given position"""
    return seq[0:pos] + consensus + seq[pos + len(consensus):len(seq)]

# Function to scramble a motif

def scramble_motif(seq):
    """Scrambles the given motif consensus sequence"""
    shuffled_motif_list = list(seq)
    random.shuffle(shuffled_motif_list)
    return ''.join(shuffled_motif_list)

# ====================================================================================================================
# Main code
# ====================================================================================================================

# Predict sequences using both models
for model_type in model_prefixes:
    model_path_prefix = model_prefixes[model_type]

    # Load keras model
    keras_model = model_from_json(open(model_path_prefix + '.json').read())
    keras_model.load_weights(model_path_prefix + '.h5')

    # Create background sequences to insert motifs into
    pred_background_activity = create_background_sequences()

    # Load the motif database for JASPAR CORE 2022 Vertebrates
    motif_df = pd.read_csv(motif_db_path, sep='\t')

    # Perform importance analysis on each motif from the database
    for index, row in motif_df.iterrows():
        motif_id = row['motif_id']
        motif_name = row['name']
        consensus = row['consensus'].upper()
        print(motif_id)

        # Create sequences by inserting the consensus motif (and scrambled) into the background sequences
        pred_sequence_activity, pred_sequence_scrambled_activity = create_sequences_with_motif(
            motif_id, consensus)

        # Create a file with all predicted activities
        create_file_for_plotting(motif_name, model_type, pred_sequence_activity,
                            pred_sequence_scrambled_activity, pred_background_activity)
