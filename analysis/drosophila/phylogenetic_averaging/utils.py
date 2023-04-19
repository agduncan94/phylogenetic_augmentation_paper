from collections import OrderedDict
from Bio import SeqIO
import numpy as np
import random

class fasta:
	"""Class for reading and operating on FASTA files"""
	
	def __init__(self, fasta_file_path):
		self.fasta_file_path = fasta_file_path
		self.fasta_dict = OrderedDict()
		self.fasta_names = []
		self.read_fasta_file(self.fasta_file_path)
		self.alphabet = "ACGT"
	
	def read_fasta_file(self, fasta_file_path):
		"""Reads a fasta file into a dictionary"""
		self.fasta_dict = OrderedDict()
		self.fasta_names = []

		with open(fasta_file_path) as handle:
			for values in SeqIO.FastaIO.SimpleFastaParser(handle):
				name = values[0]
				split_name = name.split("::")[0]
				seq = values[1].upper()
				self.fasta_dict[split_name] = [seq]
				self.fasta_names.append(split_name)
			
	def add_homolog_sequences(self, homolog_fasta):
		"""Adds homologous sequences to existing fasta object"""
		with open(homolog_fasta) as handle:
			for values in SeqIO.FastaIO.SimpleFastaParser(handle):
				name = values[0]
				split_name = name.split("::")[0]
				seq = values[1].upper()
				if 'N' not in seq:
					self.fasta_dict[split_name].append(seq)
					
	def filter_homologs(self, fold):
		"""Remove elements from the fasta names that have fewer than fold homologs"""
		indices = []
		for idx, name in enumerate(self.fasta_names):
			homologs = self.fasta_dict[name]
			if len(homologs) < fold + 1:
				indices.append(idx)
		
		remove_items_from_list_by_indices(self.fasta_names, indices)
		return indices
	
	def one_hot_encode_batch(self, indices, standardize=None, use_homologs=False, num_homologs=1):
		seqs = []
		
		# Augment sequences with homologs
		for ii in indices:
			name = self.fasta_names[ii]
			homologs = self.fasta_dict[name]
			homolog_seqs = []
			seq_id = np.random.choice(range(0, len(homologs)), 1)[0]
			
			for val in range(num_homologs):
				seq = homologs[seq_id]
				seq = self.augment_data(seq)
				homolog_seqs.append(seq)
			
			# If insufficient homologs, replace with original sequence
			#if len(homologs) >= num_homologs:
			#	seq_ids = np.random.choice(range(0, len(homologs)), num_homologs, replace=False)
			#else:
			#	seq_ids = np.random.choice(range(0, len(homologs)), len(homologs), replace=False)
			#	seqs_to_add = num_homologs - len(homologs)
			#	for val in range(1, seqs_to_add + 1):
			#		seq_ids = np.append(seq_ids, 0)
			
			#for seq_id in seq_ids:
			#	seq = homologs[seq_id]
			#	seq = self.augment_data(seq)
			#	homolog_seqs.append(seq)
			
			# One hot encode the homologs
			one_hot_data = []
			for seq in homolog_seqs:
				seq_length = len(seq)
				if (standardize is not None): 
					seq_length = int(standardize)
				one_hot_seq = np.zeros((seq_length,len(self.alphabet)))
				seq_length = min(seq_length,len(seq))
				
				for b in range(0, len(self.alphabet)):
					index = [j for j in range(0,seq_length) if seq[j] == self.alphabet[b]]
					one_hot_seq[index,b] = 1
				one_hot_data.append(one_hot_seq)
			one_hot_data = np.array(one_hot_data)
			
			seqs.append(one_hot_data)
			
		seqs = np.asarray(seqs)
		return np.nan_to_num(seqs)
	
	def augment_data(self, seq):
		k = random.randint(0, 1)
		if k == 0:
			return reverse_complement(seq)
		return seq

def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in dna[::-1]])

def count_lines_in_file(file_path):
	return sum(1 for line in open(file_path))

def remove_items_from_list_by_indices(my_list, indices):
	for index in sorted(indices, reverse=True):
		del my_list[index]
