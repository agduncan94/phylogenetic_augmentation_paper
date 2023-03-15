# Import
from keras.models import model_from_json
import pandas as pd
import numpy as np
import os

# Parameters
output_folder = "./motifs_explainn_1_homologs"
model_path = "./output/explainn_rep1/Model_explainn_rep1_homologs"
os.makedirs(output_folder, exist_ok=True)

# Load the model
keras_model_weights = model_path + '.h5'
keras_model_json = model_path + '.json'
keras_model = model_from_json(open(keras_model_json).read())
keras_model.load_weights(keras_model_weights)

# Create a probability matrix in MEME format for each motif
temp = 0.1
num_cnns = 100

for i in range(1, num_cnns + 1):
	M = keras_model.layers[i].get_weights()
	M = M[0]
	motif = []
	
	outfile = open(output_folder + '/motifs_pfm_' + str(i) + '.txt', 'w')
	outfile.write("MOTIF matrix_" + str(i) + "\n")
	outfile.write("letter-probability matrix: alength= 4 w= 19 nsites= 19\n")
	outfile.close()
	
	for n in range(len(M)):
		prob_line = np.exp(M[n]/temp)/float(4)
		prob_line = prob_line / np.sum(prob_line)
		motif.append(prob_line)
	
	pd.DataFrame(np.reshape(motif, (19, 4))).to_csv(output_folder + '/motifs_pfm_' + str(i) + '.txt', sep = '\t', index=False, header=False, mode='a')
	outfile = open(output_folder + '/motifs_pfm_' + str(i) + '.txt', 'a')
	outfile.write("\n")
	outfile.close()