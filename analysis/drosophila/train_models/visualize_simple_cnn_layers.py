# Import
from keras.models import model_from_json
import pandas as pd
import numpy as np
import os

# Parameters
output_folder = "./motifs_simple_cnn_1_n"
model_path = "./output_3/simple_cnn_rep1/Model_simple_cnn_rep1_none"
os.makedirs(output_folder, exist_ok=True)

# Load the model
keras_model_weights = model_path + '.h5'
keras_model_json = model_path + '.json'
keras_model = model_from_json(open(keras_model_json).read())
keras_model.load_weights(keras_model_weights)

# Create a probability matrix in MEME format for each motif
temp = 0.05

motifs = keras_model.layers[1].get_weights()[0]
motifs = np.reshape(motifs, (128, 19, 4))

count = 1
for motif in motifs:
	motif_len = len(motif)
	motif_arr = []
	outfile = open(output_folder + '/motifs_pfm_' + str(count) + '.txt', 'w')
	outfile.write("MOTIF matrix_" + str(count) + "\n")
	outfile.write("letter-probability matrix: alength= 4 w= 19 nsites= 19\n")
	outfile.close()
	
	for n in range(motif_len):
		prob_line = np.exp(motif[n]/temp)/float(4)
		prob_line = prob_line / np.sum(prob_line)
		motif_arr.append(prob_line)

	pd.DataFrame(motif_arr).to_csv(output_folder + '/motifs_pfm_' + str(count) + '.txt', sep = '\t', index=False, header=False, mode='a')
	outfile = open(output_folder + '/motifs_pfm_' + str(count) + '.txt', 'a')
	outfile.write("\n")
	outfile.close()
	count += 1
	