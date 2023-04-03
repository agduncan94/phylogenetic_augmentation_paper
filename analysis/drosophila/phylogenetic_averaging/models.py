# ####################################################################################################################
# models.py
#
# Class to train Keras models on Drosophila S2 data
# ####################################################################################################################

# ====================================================================================================================
# Imports
# ====================================================================================================================
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras
import keras.layers as kl
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, History
import math
import pickle
import os
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats
import os.path
import utils

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
tf.debugging.set_log_device_placement(False)
file_folder = "../process_data/output/all_orthologs/"
homolog_dir = "../process_data/output/all_orthologs/orthologs/"
output_folder = "./output/"
correlation_file_path = output_folder + 'model_correlation.tsv'

SEQUENCE_SIZE = 249
ALPHABET_SIZE = 4
BATCH_SIZE = 64
NUM_HOMOLOGS = 8

# ====================================================================================================================
# Common model code
# ====================================================================================================================
def get_batch(fasta_obj, dev_activity_array, hk_activity_array, indices, batch_size, use_homologs=False):
	"""
	Creates a batch of the input and one-hot encodes the sequences
	"""
	sequence_length = len(fasta_obj.fasta_dict[fasta_obj.fasta_names[0]][0])
	
	# One-hot encode the sequences for input as time-distributed
	X = fasta_obj.one_hot_encode_batch(indices, sequence_length, use_homologs, NUM_HOMOLOGS)
	
	# Get batch of activity values
	dev_activity_array_batch = dev_activity_array[indices]
	hk_activity_array_batch = hk_activity_array[indices]
	
	# Output values
	Y = [dev_activity_array_batch.to_numpy(), hk_activity_array_batch.to_numpy()]

	return X, Y

def data_gen(fasta_file, activity_file, homolog_dir, num_samples, shuffle_epoch_end=True, use_homologs=False, order=False):
	"""
	Generator function for loading input data in batches
	"""
	# Read fasta file
	fasta_obj = utils.fasta(fasta_file)
		
	# Add homologs
	if use_homologs:
		directory = os.fsencode(homolog_dir)
		for file in os.listdir(directory):
			filename = os.fsdecode(file)
			if filename.endswith(".fa"):
				fasta_obj.add_homolog_sequences(os.path.join(homolog_dir, filename))
	
	# Read activity file
	Activity = pd.read_table(activity_file)
	
	# Filter data to remove sequences with insufficient homologs
	#if use_homologs:
	#	filtered_indices = fasta_obj.filter_homologs(fold)
	#	Activity.drop(filtered_indices, inplace=True)
	#	Activity.reset_index(inplace=True)
		
	dev_activity_array = Activity['Dev_log2_enrichment']
	hk_activity_array = Activity['Hk_log2_enrichment']
		
	# Create the batch indices
	n_data = len(fasta_obj.fasta_names)
	if not order:
		indices = np.random.choice(list(range(n_data)), n_data, replace=False)
	else:
		indices = list(range(n_data))
	
	ii = 0
	while True:				
		yield get_batch(fasta_obj, dev_activity_array, hk_activity_array, indices[ii:ii + BATCH_SIZE], BATCH_SIZE, use_homologs)
		ii += BATCH_SIZE
		if ii >= n_data:
			ii = 0
			if shuffle_epoch_end:
				if not order:
					indices = np.random.choice(list(range(n_data)), n_data, replace=False)
				else:
					indices = list(range(n_data))
		
from keras import backend as K
def Pearson(y_true, y_pred):
	x = y_true
	y = y_pred
	mx = K.mean(x, axis=0)
	my = K.mean(y, axis=0)
	xm, ym = x - mx, y - my
	r_num = K.sum(xm * ym)
	x_square_sum = K.sum(xm * xm)
	y_square_sum = K.sum(ym * ym)
	r_den = K.sqrt(x_square_sum * y_square_sum)
	r = r_num / r_den
	return K.mean(r)

# ====================================================================================================================
# Models
# ====================================================================================================================

def ExplaiNNEncoder():
	"""Encoder for ExplaiNN from Novakosky et al"""
	# Define parameters for the encoder
	params = {
			'padding': 'same',
			'conv1_kernel_size': 19,
			'conv1_shape': 128,
			'conv1_pool_size': 10,
			'num_of_motifs': 100
		}
	
	
	# Input shape 
	input = kl.Input(shape=(SEQUENCE_SIZE, ALPHABET_SIZE))
	
	# Each CNN unit represents a motif
	encoder = []
	
	for i in range(params['num_of_motifs']):		
		# 1st convolutional layer
		cnn_x = kl.Conv1D(1, kernel_size=params['conv1_shape'], padding='same', name=str('cnn_' + str(i)))(input)
		cnn_x = BatchNormalization()(cnn_x)
		cnn_x = Activation('exponential')(cnn_x)
		cnn_x = MaxPooling1D(pool_size=7, strides=7)(cnn_x)
		cnn_x = Flatten()(cnn_x)
		
		# 1st FC layer
		cnn_x = kl.Dense(20, name=str('FC_' + str(i) + '_a'))(cnn_x)
		cnn_x = BatchNormalization()(cnn_x)
		cnn_x = Activation('relu')(cnn_x)
		cnn_x = Dropout(0.3)(cnn_x)
		
		# 2nd FC layer
		cnn_x = kl.Dense(1, name=str('FC_' + str(i) + '_b'))(cnn_x)
		cnn_x = BatchNormalization()(cnn_x)
		cnn_x = Activation('relu')(cnn_x)
		cnn_x = Flatten()(cnn_x)
		
		encoder.append(cnn_x)
		
	encoder = kl.concatenate(encoder)
	
	model = keras.models.Model([input], encoder)
	
	print(model.summary())
	
	return model

def DeepSTARREncoder():
	"""Encoder for DeepSTARR from de Almeida et al"""
	
	params = {'epochs': 100,
				  'early_stop': 10,
				  'kernel_size1': 7,
				  'kernel_size2': 3,
				  'kernel_size3': 5,
				  'kernel_size4': 3,
				  'lr': 0.002,
				  'num_filters': 256,
				  'num_filters2': 60,
				  'num_filters3': 60,
				  'num_filters4': 120,
				  'n_conv_layer': 4,
				  'n_add_layer': 2,
				  'dropout_prob': 0.4,
				  'dense_neurons1': 256,
				  'dense_neurons2': 256,
				  'pad':'same'}
	
	# Input shape
	input = kl.Input(shape=(SEQUENCE_SIZE, ALPHABET_SIZE))
	
	# Define encoder to create embedding vector
	x = kl.Conv1D(params['num_filters'], kernel_size=params['kernel_size1'],
				  padding=params['pad'],
				  name='Conv1D_1st')(input)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling1D(2)(x)

	for i in range(1, params['n_conv_layer']):
		x = kl.Conv1D(params['num_filters'+str(i+1)],
					  kernel_size=params['kernel_size'+str(i+1)],
					  padding=params['pad'],
					  name=str('Conv1D_'+str(i+1)))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling1D(2)(x)
	
	x = Flatten()(x)
	
	for i in range(0, params['n_add_layer']):
		x = kl.Dense(params['dense_neurons'+str(i+1)],
					 name=str('Dense_'+str(i+1)))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Dropout(params['dropout_prob'])(x)
	encoder = x

	model = keras.models.Model([input], encoder)
	
	print(model.summary())

	return model

def MotifEncoder():
	"""Encoder for a motif model"""
	
	# Define parameters for the encoder
	params = {
			'padding': 'same',
			'conv1_kernel_size': 19,
			'conv1_shape': 128,
			'conv1_pool_size': 10
		}
	
	# Input shape 
	input = kl.Input(shape=(SEQUENCE_SIZE, ALPHABET_SIZE))
	
	# Define encoder to create embedding vector
	encoder = kl.Conv1D(params['conv1_shape'], kernel_size=params['conv1_kernel_size'],
				  padding=params['padding'],
				  name='Conv1D', input_shape=(SEQUENCE_SIZE, ALPHABET_SIZE))(input)
	encoder = BatchNormalization()(encoder)
	encoder = Activation('exponential')(encoder)
	encoder = GlobalMaxPooling1D()(encoder)
	encoder = Flatten()(encoder)
	
	model = keras.models.Model([input], encoder)
	
	print(model.summary())
	
	return model

def MotifDeepSTARREncoder():
	"""Encoder for a model like DeepSTARR, but with an interpretable motif layer"""
	
	# Define parameters for the encoder
	params = {
			'padding': 'same',
			'conv1_kernel_size': 19,
			'conv1_shape': 128,
			'conv1_pool_size': 10
		}
	
	# Input shape 
	input = kl.Input(shape=(SEQUENCE_SIZE, ALPHABET_SIZE))
	
	# Define encoder to create embedding vector
	encoder = kl.Conv1D(params['conv1_shape'], kernel_size=params['conv1_kernel_size'],
				  padding=params['padding'],
				  name='Conv1D', input_shape=(SEQUENCE_SIZE, ALPHABET_SIZE))(input)
	encoder = BatchNormalization()(encoder)
	encoder = Activation('exponential')(encoder)
	encoder = MaxPooling1D(10)(encoder)
	encoder = Flatten()(encoder)
	
	# First dense layer
	encoder = kl.Dense(128, name='Dense_a')(encoder)
	encoder = BatchNormalization()(encoder)
	encoder = Activation('relu')(encoder)
	encoder = Dropout(0.3)(encoder)
	
	# Second dense layer
	encoder = kl.Dense(128, name='Dense_b')(encoder)
	encoder = BatchNormalization()(encoder)
	encoder = Activation('relu')(encoder)
	encoder = Dropout(0.3)(encoder)
	
	model = keras.models.Model([input], encoder)
	
	print(model.summary())
	
	return model
	
def CommonModel(encoder, pooling="average"):
	"""Common head for encoders"""
	# Define parameters for the model
	params = {
			'lr': 0.002
		}
	
	# Define model
	input = kl.Input(shape=(NUM_HOMOLOGS, SEQUENCE_SIZE, ALPHABET_SIZE))
	td_layer = tf.keras.layers.TimeDistributed(encoder)(input)
	
	# Merge the homolog representations
	if pooling == 'average':
		pool_layer = GlobalAveragePooling1D()(td_layer)
	else:
		pool_layer = GlobalMaxPooling1D()(td_layer)
	bottleneck = pool_layer
	
	# heads per task (Dev and Hk enhancer activities)
	tasks = ['Dev', 'Hk']
	outputs = []
	for task in tasks:
		outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))

	model = keras.models.Model([input], outputs)
	model.compile(keras.optimizers.Adam(learning_rate=params['lr']),
				  loss=['mse', 'mse'],
				  loss_weights=[1, 1],
				  metrics=[Pearson])
	
	print(model.summary())
	
	return model

		

# ====================================================================================================================
# Train models
# ====================================================================================================================
def train(model, model_type, use_homologs, replicate):
	# Parameters
	epochs = 100
	early_stop = 10

	# Create a unique identifier
	model_id = model_type + "_rep" + str(replicate)
	
	# Create folder for output
	model_output_folder = output_folder + model_id + "/"
	os.makedirs(model_output_folder, exist_ok=True)
	
	# Determine the number of sequences in training and validation (For generator)
	num_samples_train = utils.count_lines_in_file(file_folder + "Sequences_activity_Train.txt") - 1
	num_samples_val = utils.count_lines_in_file(file_folder + "Sequences_activity_Val.txt") - 1
	num_samples_test = utils.count_lines_in_file(file_folder + "Sequences_activity_Test.txt") - 1

	# Data for train and validation sets (TODO: Homolog dir needs subdirectory for type)
	datagen_train = data_gen(file_folder + "Sequences_Train.fa", file_folder + "Sequences_activity_Train.txt", homolog_dir + 'Sequences_Train/', num_samples_train, True, use_homologs)
	datagen_val = data_gen(file_folder + "Sequences_Val.fa", file_folder + "Sequences_activity_Val.txt", homolog_dir + 'Sequences_Val/', num_samples_val, True, use_homologs)

	# Fit model
	history=model.fit(datagen_train,
								  validation_data=datagen_val,
								  epochs=epochs,
								  steps_per_epoch=math.ceil(num_samples_train / BATCH_SIZE),
								  validation_steps=math.ceil(num_samples_val / BATCH_SIZE),
								  callbacks=[EarlyStopping(patience=early_stop, monitor="val_loss", restore_best_weights=True),
											 History()])
	
	# Save results
	epochs_total = len(history.history['val_Dense_Dev_Pearson'])
	if use_homologs:
		save_model(model_id + "_homologs", model, history, model_output_folder)
		test_correlation_dev, test_correlation_hk = plot_prediction_vs_actual(model, file_folder + "Sequences_Test.fa", file_folder + "Sequences_activity_Test.txt", model_output_folder + 'Model_' + model_id + "_homologs_Test", num_samples_test, homolog_dir + 'Sequences_Test/')
		write_to_file(model_id + "\thomologs\t" + model_type + "\t" + str(replicate) + "\t" + str(history.history['Dense_Dev_Pearson'][epochs_total-1]) + "\t" + str(history.history['Dense_Hk_Pearson'][epochs_total-1]) + "\t" + str(history.history['val_Dense_Dev_Pearson'][epochs_total-1]) + "\t" + str(history.history['val_Dense_Hk_Pearson'][epochs_total-1]) + "\t" + str(test_correlation_dev) + "\t" + str(test_correlation_hk) + "\n")
	else:
		save_model(model_id + "_none", model, history, model_output_folder)
		test_correlation_dev, test_correlation_hk = plot_prediction_vs_actual(model, file_folder + "Sequences_Test.fa", file_folder + "Sequences_activity_Test.txt", model_output_folder + 'Model_' + model_id + "_none_Test", num_samples_test, homolog_dir + 'Sequences_Test/')
		write_to_file(model_id + "\tnone\t" + model_type + "\t" + str(replicate) + "\t" + str(history.history['Dense_Dev_Pearson'][epochs_total-1]) + "\t" + str(history.history['Dense_Hk_Pearson'][epochs_total-1]) + "\t" + str(history.history['val_Dense_Dev_Pearson'][epochs_total-1]) + "\t" + str(history.history['val_Dense_Hk_Pearson'][epochs_total-1]) + "\t" + str(test_correlation_dev) + "\t" + str(test_correlation_hk) + "\n")
		
	# Save plots for performance and loss
	if use_homologs:
		plot_scatterplots(history, model_output_folder, model_id, 'homologs')
	else:
		plot_scatterplots(history, model_output_folder, model_id, 'none')
		
	# Save the model and history
	model_json = model.to_json()
	with open(model_output_folder + 'Model_' + model_id + '.json', "w") as json_file:
		json_file.write(model_json)
		
	model.save_weights(model_output_folder + 'Model_' + model_id + '.h5')
	
	with open(model_output_folder + 'Model_' + model_id + '_history', 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	
def train_deepstarr(use_homologs, replicate, model_type="DeepSTARR"):
	encoder = DeepSTARREncoder()
	model = CommonModel(encoder, 'average')
	train(model, model_type, use_homologs, replicate)

def train_explainn(use_homologs, replicate, model_type="ExplaiNN"):
	encoder = ExplaiNNEncoder()
	model = CommonModel(encoder, 'average')
	train(model, model_type, use_homologs, replicate)

def train_motif_avg_model(use_homologs, replicate, model_type="MotifAvg"):
	encoder = MotifEncoder()
	model = CommonModel(encoder, 'average')
	train(model, model_type, use_homologs, replicate)
	
def train_motif_max_model(use_homologs, replicate, model_type="MotifMax"):
	encoder = MotifEncoder()
	model = CommonModel(encoder, 'max')
	train(model, model_type, use_homologs, replicate)
	
def train_motif_deepstarr(use_homologs, replicate, model_type="MotifDeepSTARR"):
	encoder = MotifDeepSTARREncoder()
	model = CommonModel(encoder, 'average')
	train(model, model_type, use_homologs, replicate)
	
# ====================================================================================================================
# Plot model performance
# ====================================================================================================================
def plot_prediction_vs_actual(model, fasta_file, activity_file, output_file_prefix, num_samples, homolog_dir):
	# Load the activity data
	Y_0 = np.array([])
	Y_1 = np.array([])
	
	count = 0
	for x,y in data_gen(fasta_file, activity_file, homolog_dir, num_samples, use_homologs=True, order=True):
		Y_0 = np.concatenate((Y_0, y[0]), axis=0)
		Y_1 = np.concatenate((Y_1, y[1]), axis=0)
		count += 1
		if count > (num_samples / BATCH_SIZE):
			break
		
	Y = [Y_0, Y_1]
	
	# Get model predictions
	data_generator = data_gen(fasta_file, activity_file, homolog_dir, num_samples, use_homologs=True, order=True)
	Y_pred = model.predict(data_generator, steps=math.ceil(num_samples / BATCH_SIZE))
	
	# Calculate PCC
	correlation_dev = stats.pearsonr(Y[0], Y_pred[0].squeeze())[0]
	correlation_hk = stats.pearsonr(Y[1], Y_pred[1].squeeze())[0]
	
	# Plot 2iL correlation
	fig, ax = plt.subplots()
	ax.scatter(Y[0], Y_pred[0].squeeze())
	ax.set_title("Dev Correlation")
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	at = AnchoredText("PCC:" + str(correlation_dev), prop=dict(size=15), frameon=True, loc='upper left')
	at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
	ax.add_artist(at)
	plt.savefig(output_file_prefix + '_Dev_correlation.png')
	plt.clf()
	
	# Plot SL correlation
	fig, ax = plt.subplots()
	ax.scatter(Y[1], Y_pred[1].squeeze())
	ax.set_title("Hk Correlation")
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	at = AnchoredText("PCC:" + str(correlation_hk), prop=dict(size=15), frameon=True, loc='upper left')
	at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
	ax.add_artist(at)
	plt.savefig(output_file_prefix + '_Hk_correlation.png')
	plt.clf()
	
	return correlation_dev, correlation_hk
	

def plot_scatterplot(history, a, b, x, y, title, filename):
	plt.plot(history.history[a])
	plt.plot(history.history[b])
	plt.title(title)
	plt.ylabel(x)
	plt.xlabel(y)
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(filename)
	plt.clf()
	 
def plot_scatterplots(history, model_output_folder, model_id, name):	
	plot_scatterplot(history, 'Dense_Dev_Pearson', 'val_Dense_Dev_Pearson', 'PCC', 'epoch', 'Model performance Dev (Pearson)', model_output_folder + 'Model_' + model_id + '_' + name + '_Dev_pearson.png')
	plot_scatterplot(history, 'Dense_Hk_Pearson', 'val_Dense_Hk_Pearson', 'PCC', 'epoch', 'Model performance Hk (Pearson)', model_output_folder + 'Model_' + model_id + '_' + name + '_Hk_pearson.png')

	plot_scatterplot(history, 'Dense_Dev_loss', 'val_Dense_Dev_loss', 'loss', 'epoch', 'Model loss Dev', model_output_folder + 'Model_' + model_id + '_' + name + '_Dev_loss.png')		
	plot_scatterplot(history, 'Dense_Hk_loss', 'val_Dense_Hk_loss', 'loss', 'epoch', 'Model loss Hk', model_output_folder + 'Model_' + model_id + '_' + name + '_Hk_loss.png')
	
# ====================================================================================================================
# Helpers
# ====================================================================================================================
def write_to_file(line):
	if os.path.isfile(correlation_file_path):
		f = open(correlation_file_path, "a")
		f.write(line)
		f.close()
	else:
		f = open(correlation_file_path, "w")
		f.write("name\ttype\tmodel\treplicate\tpcc_train_dev\tpcc_train_hk\tpcc_val_dev\tpcc_val_hk\tpcc_test_dev\tpcc_test_hk\n")
		f.write(line)
		f.close()

def save_model(model_name, model, history, model_output_folder):
	# Save the model and history
	model_json = model.to_json()
	with open(model_output_folder + 'Model_' + model_name + '.json', "w") as json_file:
		json_file.write(model_json)
		
	model.save_weights(model_output_folder + 'Model_' + model_name + '.h5')
	
	with open(model_output_folder + 'Model_' + model_name + '_history', 'wb') as file_pi:
		pickle.dump(history.history, file_pi)