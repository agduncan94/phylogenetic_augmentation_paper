# Import
from keras.models import model_from_json
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers import BatchNormalization, InputLayer, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import logomaker as lm

# Load the model

model_path = "./output_relu/simple_cnn_rep1/Model_simple_cnn_rep1_homologs_finetune"
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
	outfile = open('./motifs_temp_0.05/motifs_pfm_' + str(count) + '.txt', 'w')
	outfile.write("MOTIF matrix_" + str(count) + " length=" + str(motif_len) + "\n")
	outfile.write("letter-probability matrix: alength= 4 w= 19 nsites= 19\n")
	outfile.close()
	
	for n in range(motif_len):
		prob_line = np.exp(motif[n]/temp)/float(4)
		prob_line = prob_line / np.sum(prob_line)
		motif_arr.append(prob_line)

	pd.DataFrame(motif_arr).to_csv('./motifs_temp_0.05/motifs_pfm_' + str(count) + '.txt', sep = '\t', index=False, header=False, mode='a')
	outfile = open('./motifs_temp_0.05/motifs_pfm_' + str(count) + '.txt', 'a')
	outfile.write("\n")
	outfile.close()
	count += 1
	