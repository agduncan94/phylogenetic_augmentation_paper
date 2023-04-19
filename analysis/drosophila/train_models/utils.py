from collections import OrderedDict
from Bio import SeqIO, motifs
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
		self.fasta_dict = OrderedDict()
		self.fasta_names = []
		with open(fasta_file_path) as handle:
			for values in SeqIO.FastaIO.SimpleFastaParser(handle):
				name = values[0]
				split_name = name.split("::")[0]
				seq = values[1].upper()
				self.fasta_dict[split_name] = [seq]
				self.fasta_names.append(split_name)
				
	def sample_fasta(self, indices):
		# need to update fasta_names and fasta_dict
		new_fasta_names = []
		new_fasta_dict = OrderedDict()
		for i in indices:
			name = self.fasta_names[i]
			new_fasta_names.append(name)
			new_fasta_dict[name] = self.fasta_dict[name]
		self.fasta_names = new_fasta_names
		self.fasta_dict = new_fasta_dict
			
	def add_homolog_sequences(self, homolog_fasta):
		with open(homolog_fasta) as handle:
			for values in SeqIO.FastaIO.SimpleFastaParser(handle):
				name = values[0]
				split_name = name.split("::")[0]
				seq = values[1].upper()
				if 'N' not in seq:
					self.fasta_dict[split_name].append(seq)
	
	def one_hot_encode_batch(self, indices, standardize=None, use_homologs=False, fold=1):
		seqs = []
		seq_multiplier = []
		
		# Augment sequences with homologs and reverse complement
		for ii in indices:
			name = self.fasta_names[ii]
			homologs = self.fasta_dict[name]
			num_homologs = len(homologs) - 1

			if use_homologs:
				homologs_to_add = min(fold - 1, num_homologs)
				seq_multiplier.append(homologs_to_add + 1)
				
				# Sample from homologs
				seq_ids = np.random.choice(range(1, len(homologs)), homologs_to_add, replace=False)
				
				for seq_id in seq_ids:
					seq = homologs[seq_id]
					seq = self.augment_data(seq)
					seqs.append(seq)
			else:
				seq_multiplier.append(1)
				
			# Add the real sequence
			seq = self.fasta_dict[name][0]
			seq = self.augment_data(seq)
			seqs.append(seq)
		
		# One hot encode the sequences in the batch
		one_hot_data = []
		for seq in seqs:
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
		return one_hot_data, seq_multiplier

	def augment_data(self, seq):
		k = random.randint(0, 1)
		if k == 0:
			return reverse_complement(seq)
		return seq
	
class motif_db:
	"""Class for PWM files"""
	
	def __init__(self, pfm_file_path):
		self.pfm_file_path = pfm_file_path
		self.motif_db = {}
		self.read_pfm_db(self.pfm_file_path)
		
	def read_pfm_db(self, pfm_file_path):
	    self.motif_db = {}
	    fh = open(pfm_file_path)
	    for m in motifs.parse(fh, "jaspar"):
	        self.motif_db[m.matrix_id] = m
	

def convert_pwms_to_filter(motif):
	pwm = motif.counts.normalize(1).log_odds()
	pwm_np = np.array([pwm['A'], pwm['C'], pwm['G'], pwm['T']])
	return pwm_np.T

def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in dna[::-1]])

def count_lines_in_file(file_path):
	return sum(1 for line in open(file_path))
