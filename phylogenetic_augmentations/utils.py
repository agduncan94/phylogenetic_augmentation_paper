# ####################################################################################################################
# utils.py
#
# Utility classes and functions for model training
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
from collections import OrderedDict
from Bio import SeqIO
import numpy as np
import random
import h5py

# ====================================================================================================================
# Homolog FASTA class
# ====================================================================================================================


class homolog_fastas:
    """Class for reading and operating on FASTA files to store homolog information. Used for the Drosophila data."""

    def __init__(self, fasta_file_path):
        """Initialize object with FASTA file information from the target species"""
        self.fasta_file_path = fasta_file_path
        self.fasta_dict = OrderedDict()
        self.fasta_names = []
        self.read_fasta_file(self.fasta_file_path)
        self.alphabet = "ACGT"

    def read_fasta_file(self, input_df):
        """Read a FASTA file and store information into object"""
        self.fasta_dict = OrderedDict()
        self.fasta_names = []

        for index, row in input_df.iterrows():
            name = row['Name']
            split_name = name.split("::")[0]
            seq = row['Seq']
            self.fasta_dict[split_name] = [seq]
            self.fasta_names.append(split_name)

    def sample_fasta(self, indices):
        """Samples a set of indices from the FASTA object"""
        new_fasta_names = []
        new_fasta_dict = OrderedDict()
        for i in indices:
            name = self.fasta_names[i]
            new_fasta_names.append(name)
            new_fasta_dict[name] = self.fasta_dict[name]
        self.fasta_names = new_fasta_names
        self.fasta_dict = new_fasta_dict

    def add_homolog_sequences(self, homolog_fasta):
        """Add homolog information from an addtional FASTA file to the FASTA object"""
        with open(homolog_fasta) as handle:
            for values in SeqIO.FastaIO.SimpleFastaParser(handle):
                name = values[0]
                split_name = name.split("::")[0]
                seq = values[1].upper()
                if 'N' not in seq and split_name in self.fasta_dict:
                    self.fasta_dict[split_name].append(seq)

    def one_hot_encode_batch(self, indices, standardize=None, use_homologs=False, homolog_rate=1.0, rev_comp=True):
        """One hot encode a batch of sequences"""
        seqs = []

        # Augment sequences with homologs and reverse complement
        for ii in indices:
            name = self.fasta_names[ii]
            homologs = self.fasta_dict[name]

            if use_homologs:
                # Apply phylo aug at a given rate
                seq_id = 0
                if random.choices(population=[0, 1], weights=[1 - homolog_rate, homolog_rate])[0] == 1:
                    # Sample from homologs
                    seq_id = np.random.randint(0, len(homologs))
                seq = homologs[seq_id]
                if rev_comp:
                    seq = self.rev_comp_augmentation(seq)
                seqs.append(seq)
            else:
                seq = homologs[0]
                if rev_comp:
                    seq = self.rev_comp_augmentation(seq)
                seqs.append(seq)

        # One hot encode the sequences in the batch
        one_hot_data = []
        for seq in seqs:
            seq_length = len(seq)
            if (standardize is not None):
                seq_length = int(standardize)
            one_hot_seq = np.zeros((seq_length, len(self.alphabet)))
            seq_length = min(seq_length, len(seq))

            for b in range(0, len(self.alphabet)):
                index = [j for j in range(
                    0, seq_length) if seq[j] == self.alphabet[b]]
                one_hot_seq[index, b] = 1
            one_hot_data.append(one_hot_seq)
        one_hot_data = np.array(one_hot_data)

        X = np.nan_to_num(one_hot_data)
        X_batch = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

        return X_batch

    def rev_comp_augmentation(self, seq):
        """Apply reverse complement randomly to input sequence"""
        return seq if random.randint(0, 1) == 0 else reverse_complement(seq)

# ====================================================================================================================
# Utility functions
# ====================================================================================================================


def one_hot_encode_batch_hdf5(split_type, hdf5_file, seq_ids, standardize=None, use_homologs=False):
    """Class for encoding batches from an hdf5 file. Used for the Basset data."""
    seqs = []
    alphabet = "ACGT"
    with h5py.File(hdf5_file, "r") as f:
        # Augment sequences with homologs and reverse complement
        for seq_id in seq_ids:
            if use_homologs:
                # Sample from homologs
                homologs = f[split_type + '/sequences/' + seq_id]
                seq_pos = np.random.randint(0, homologs[:].shape[0])
                seq = homologs[seq_pos]
                seqs.append(seq)
            else:
                homologs = f[split_type + '/sequences/' + seq_id]
                seq = homologs[0]
                seqs.append(seq)

        # print(np.array(seqs).shape)

        # One hot encode the sequences in the batch
        one_hot_data = []
        for seq in seqs:
            one_hot_data.append(seq.astype(float))
        one_hot_data = np.array(one_hot_data)

    X = np.nan_to_num(one_hot_data)
    X_batch = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    return X_batch


def reverse_complement(dna):
    """Performs reverse complement on the given DNA sequence"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in dna[::-1]])


def count_lines_in_file(file_path):
    """Counts the number of lines in a file"""
    return sum(1 for line in open(file_path))
