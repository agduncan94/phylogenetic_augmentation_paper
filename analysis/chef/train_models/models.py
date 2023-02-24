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
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import BatchNormalization
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
file_folder = "../process_data/output/"
homolog_dir = "../process_data/output/orthologs/"
output_folder = "./output/"
correlation_file_path = output_folder + 'model_correlation.tsv'
batch_size = 32
fold = 4

# ====================================================================================================================
# Common model code
# ====================================================================================================================
def get_batch(fasta_obj, h3k27ac_array, tf_sum_array, indices, batch_size, use_homologs=False, fold=1):
	"""
	Creates a batch of the input and one-hot encodes the sequences
	"""
	sequence_length = len(fasta_obj.fasta_dict[fasta_obj.fasta_names[0]][0])
	
	# One-hot encode the sequences
	seqs, seq_multiplier = fasta_obj.one_hot_encode_batch(indices, sequence_length, use_homologs, fold)
	X = np.nan_to_num(seqs)
	X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
	
	# Get batch of activity values
	h3k27ac_array_batch = h3k27ac_array[indices]
	tf_sum_array_batch = tf_sum_array[indices]
	
	if use_homologs:
		Y = [np.repeat(h3k27ac_array_batch.to_numpy(), seq_multiplier), np.repeat(tf_sum_array_batch.to_numpy(), seq_multiplier)]
	else:
		Y = [h3k27ac_array_batch.to_numpy(), tf_sum_array_batch.to_numpy()]
	
	return X_reshaped, Y

def data_gen(fasta_file, activity_file, homolog_dir, num_samples, batch_size, shuffle_epoch_end=True, use_homologs=False, fold=1, order=False):
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
	
	h3k27ac_array = Activity['h3k27ac_log2_enrichment']
	tf_sum_array = Activity['tf_sum']
	
	# Create the batch indices
	n_data = len(fasta_obj.fasta_names)
	if not order:
		indices = np.random.choice(list(range(num_samples)), num_samples, replace=False)
	else:
		indices = list(range(n_data))
	
	ii = 0
	while True:		
		# Calculate how large a batch size to make
		if use_homologs:
			new_batch_size = calculate_batch_size(fasta_obj, indices, batch_size, ii, fold)
		else:
			new_batch_size = batch_size
		
		yield get_batch(fasta_obj, h3k27ac_array, tf_sum_array, indices[ii:ii + new_batch_size], new_batch_size, use_homologs, fold)
		ii += new_batch_size
		if ii >= num_samples:
			ii = 0
			if shuffle_epoch_end:
				if not order:
					indices = np.random.choice(list(range(num_samples)), num_samples, replace=False)
				else:
					indices = list(range(n_data))

def calculate_batch_size(fasta_obj, indices, batch_size, ii, fold):
	"""
	Determines new batch size for getting appropriate number of sequences
	"""
	goal_sequence_size = batch_size * fold
	current_sequence_size = 0
	new_batch_size = 0
	
	for i in range(ii, len(indices)):
		name = fasta_obj.fasta_names[i]
		homologs = fasta_obj.fasta_dict[name]
		num_homologs = len(homologs) - 1
		
		current_sequence_size += min(fold, num_homologs)
		new_batch_size += 1
		
		if current_sequence_size >= goal_sequence_size:
			break
		
	return new_batch_size
		
from scipy.stats import spearmanr
def Spearman(y_true, y_pred):
     return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout = tf.float32))
 
 
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

def DeepSTARR():
    """
	DeepSTARR architecture
    """
    lr = params['lr']
    dropout_prob = params['dropout_prob']
    n_conv_layer = params['n_conv_layer']
    n_add_layer = params['n_add_layer']
    
    input = kl.Input(shape=(700, 4))
    x = kl.Conv1D(params['num_filters'], kernel_size=params['kernel_size1'],
                  padding=params['pad'],
                  name='Conv1D_1st')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)

    for i in range(1, n_conv_layer):
        x = kl.Conv1D(params['num_filters'+str(i+1)],
                      kernel_size=params['kernel_size'+str(i+1)],
                      padding=params['pad'],
                      name=str('Conv1D_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
    
    x = Flatten()(x)
    
    # dense layers
    for i in range(0, n_add_layer):
        x = kl.Dense(params['dense_neurons'+str(i+1)],
                     name=str('Dense_'+str(i+1)))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_prob)(x)
    bottleneck = x
    
    # heads per task (h3k27ac and tf_sum enhancer activities)
    tasks = ['h3k27ac', 'tf_sum']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))

    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=lr),
                  loss=['mse', 'mse'], # loss
                  loss_weights=[1, 1], # loss weigths to balance
                  metrics=[Spearman, Pearson]) # additional track metric

    return model

def ExplaiNN():
    """
    ExplaiNN architecture from Novakosky et al
    """
    input = kl.Input(shape=(700, 4))
	
    # Each CNN unit represents a motif
    cnns = []
	
    for i in range(100):		
		# 1st convolutional layer
        cnn_x = kl.Conv1D(1, kernel_size=19, padding='same', name=str('cnn_' + str(i)))(input)
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
		
        cnns.append(cnn_x)
		
    cnns = kl.concatenate(cnns)
            
    tasks = ['h3k27ac', 'tf_sum']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(cnns))

    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=0.002),
                  loss=['mse', 'mse'],
                  loss_weights=[1, 1],
                  metrics=[Spearman, Pearson])

    return model


def SimpleModel():
    """
    Simple architecture 
    """
    lr = 0.002
	
    input = kl.Input(shape=(700, 4))
	
	# Convolutional layer
    x = kl.Conv1D(128, kernel_size=19,
                  padding=params['pad'],
                  name='Conv1D')(input)
    x = BatchNormalization()(x)
    x = Activation('exponential')(x)
    x = MaxPooling1D(10)(x)
    
    x = Flatten()(x)
    
    # First dense layer
    x = kl.Dense(128, name='Dense_a')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
	
    # Second dense layer
    x = kl.Dense(128, name='Dense_b')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
	
    bottleneck = x
    
    # heads per task
    tasks = ['h3k27ac', 'tf_sum']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))

    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=lr),
                  loss=['mse', 'mse'], # loss
                  loss_weights=[1, 1], # loss weigths to balance
                  metrics=[Spearman, Pearson]) # additional track metric

    return model

# ====================================================================================================================
# Train models
# ====================================================================================================================
def train(model, model_type, use_homologs, replicate):
	# Parameters
	epochs = 100
	fine_tune_epochs = 5
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

	# Data for train and validation sets
	datagen_train = data_gen(file_folder + "Sequences_Train.fa", file_folder + "Sequences_activity_Train.txt", homolog_dir, num_samples_train, batch_size, True, use_homologs, fold)
	datagen_val = data_gen(file_folder + "Sequences_Val.fa", file_folder + "Sequences_activity_Val.txt", homolog_dir, num_samples_val, batch_size, True, False, fold)

	# Fit model
	history=model.fit(datagen_train,
                                  validation_data=datagen_val,
								  epochs=epochs,
								  steps_per_epoch=math.ceil(num_samples_train / batch_size),
								  validation_steps=math.ceil(num_samples_val / batch_size),
                                  callbacks=[EarlyStopping(patience=early_stop, monitor="val_loss", restore_best_weights=True),
                                             History()])
	
	# Save results
	epochs_total = len(history.history['val_Dense_h3k27ac_Spearman'])
	if use_homologs:
		save_model(model_id + "_homologs", model, history, model_output_folder)
		test_correlation_h3k27ac, test_correlation_tf_sum = plot_prediction_vs_actual(model, file_folder + "Sequences_Test.fa", file_folder + "Sequences_activity_Test.txt", model_output_folder + 'Model_' + model_id + "_homologs_Test", num_samples_test)
		write_to_file(model_id + "\thomologs\t" + model_type + "\t" + str(replicate) + "\t" + str(history.history['Dense_h3k27ac_Pearson'][epochs_total-1]) + "\t" + str(history.history['Dense_tf_sum_Pearson'][epochs_total-1]) + "\t" + str(history.history['val_Dense_h3k27ac_Pearson'][epochs_total-1]) + "\t" + str(history.history['val_Dense_tf_sum_Pearson'][epochs_total-1]) + "\t" + str(test_correlation_h3k27ac) + "\t" + str(test_correlation_tf_sum) + "\n")
	else:
		save_model(model_id + "_none", model, history, model_output_folder)
		test_correlation_h3k27ac, test_correlation_tf_sum = plot_prediction_vs_actual(model, file_folder + "Sequences_Test.fa", file_folder + "Sequences_activity_Test.txt", model_output_folder + 'Model_' + model_id + "_none_Test", num_samples_test)
		write_to_file(model_id + "\tnone\t" + model_type + "\t" + str(replicate) + "\t" + str(history.history['Dense_h3k27ac_Pearson'][epochs_total-1]) + "\t" + str(history.history['Dense_tf_sum_Pearson'][epochs_total-1]) + "\t" + str(history.history['val_Dense_h3k27ac_Pearson'][epochs_total-1]) + "\t" + str(history.history['val_Dense_tf_sum_Pearson'][epochs_total-1]) + "\t" + str(test_correlation_h3k27ac) + "\t" + str(test_correlation_tf_sum) + "\n")
	
	# Save plots for performance and loss
	if use_homologs:
		plot_scatterplots(history, model_output_folder, model_id, 'homologs')
	else:
		plot_scatterplots(history, model_output_folder, model_id, 'none')
		
	# Fine tuning on original training
	model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
			   loss=['mse', 'mse'],
			   loss_weights=[1,1],
			   metrics=[Spearman, Pearson])
	
	if use_homologs:
		datagen_train = data_gen(file_folder + "Sequences_Train.fa", file_folder + "Sequences_activity_Train.txt", homolog_dir, num_samples_train, batch_size, True, False, fold)
	
	fine_tune_history = model.fit(datagen_train,
							   validation_data=datagen_val,
							   steps_per_epoch=math.ceil(num_samples_train / batch_size),
							   validation_steps=math.ceil(num_samples_val / batch_size),
							   epochs=fine_tune_epochs)
	
	# Save results
	if use_homologs:
		save_model(model_id + "_homologs_finetune", model, fine_tune_history, model_output_folder)
		test_correlation_h3k27ac, test_correlation_tf_sum = plot_prediction_vs_actual(model, file_folder + "Sequences_Test.fa", file_folder + "Sequences_activity_Test.txt", model_output_folder + 'Model_' + model_id + "_homologs_finetune_Test", num_samples_test)
		write_to_file(model_id + "\thomologs_finetune\t" + model_type + "\t" + str(replicate) + "\t" + str(fine_tune_history.history['Dense_h3k27ac_Pearson'][fine_tune_epochs-1]) + "\t" + str(fine_tune_history.history['Dense_tf_sum_Pearson'][fine_tune_epochs-1]) + "\t" + str(fine_tune_history.history['val_Dense_h3k27ac_Pearson'][fine_tune_epochs-1]) + "\t" + str(fine_tune_history.history['val_Dense_tf_sum_Pearson'][fine_tune_epochs-1]) + "\t" + str(test_correlation_h3k27ac) + "\t" + str(test_correlation_tf_sum) + "\n")
	else:
		save_model(model_id + "_finetune", model, fine_tune_history, model_output_folder)
		test_correlation_h3k27ac, test_correlation_tf_sum = plot_prediction_vs_actual(model, file_folder + "Sequences_Test.fa", file_folder + "Sequences_activity_Test.txt", model_output_folder + 'Model_' + model_id + "_finetune_Test", num_samples_test)
		write_to_file(model_id + "\tfinetune\t" + model_type + "\t" + str(replicate) + "\t" + str(fine_tune_history.history['Dense_h3k27ac_Pearson'][fine_tune_epochs-1]) + "\t" + str(fine_tune_history.history['Dense_tf_sum_Pearson'][fine_tune_epochs-1]) + "\t" + str(fine_tune_history.history['val_Dense_h3k27ac_Pearson'][fine_tune_epochs-1]) + "\t" + str(fine_tune_history.history['val_Dense_tf_sum_Pearson'][fine_tune_epochs-1]) + "\t" + str(test_correlation_h3k27ac) + "\t" + str(test_correlation_tf_sum) + "\n")
	
	# Save the model and history
	model_json = model.to_json()
	with open(model_output_folder + 'Model_' + model_id + '.json', "w") as json_file:
	    json_file.write(model_json)
		
	model.save_weights(model_output_folder + 'Model_' + model_id + '.h5')
	
	with open(model_output_folder + 'Model_' + model_id + '_history', 'wb') as file_pi:
	    pickle.dump(history.history, file_pi)
    	
	# Save plots for performance and loss
	if use_homologs:
		plot_scatterplots(fine_tune_history, model_output_folder, model_id, 'homologs_finetune')
	else:
		plot_scatterplots(fine_tune_history, model_output_folder, model_id, 'finetune')
	
def train_deepstarr(use_homologs, replicate, model_type="DeepSTARR"):
	model = DeepSTARR()
	train(model, model_type, use_homologs, replicate)

def train_explainn(use_homologs, replicate, model_type="ExplaiNN"):
	model = ExplaiNN()
	train(model, model_type, use_homologs, replicate)

def train_simple_model(use_homologs, replicate, model_type="SimpleCnn"):
	model = SimpleModel()
	train(model, model_type, use_homologs, replicate)
	
# ====================================================================================================================
# Plot model performance
# ====================================================================================================================
def plot_prediction_vs_actual(model, fasta_file, activity_file, output_file_prefix, num_samples):
	# Load the activity data
	Y_0 = np.array([])
	Y_1 = np.array([])
	
	count = 0
	for x,y in data_gen(fasta_file, activity_file, homolog_dir, num_samples, batch_size, use_homologs=False, order=True):
		Y_0 = np.concatenate((Y_0, y[0]), axis=0)
		Y_1 = np.concatenate((Y_1, y[1]), axis=0)
		count += 1
		if count > (num_samples / batch_size):
			break
		
	Y = [Y_0, Y_1]
	
	# Get model predictions
	data_generator = data_gen(fasta_file, activity_file, homolog_dir, num_samples, batch_size, use_homologs=False, order=True)
	Y_pred = model.predict(data_generator, steps=math.ceil(num_samples / batch_size))
	
	# Calculate PCC
	correlation_h3k27ac = stats.pearsonr(Y[0], Y_pred[0].squeeze())[0]
	correlation_tf_sum = stats.pearsonr(Y[1], Y_pred[1].squeeze())[0]
	
	# Plot h3k27ac correlation
	fig, ax = plt.subplots()
	ax.scatter(Y[0], Y_pred[0].squeeze())
	ax.set_title("h3k27ac Correlation")
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	at = AnchoredText("PCC:" + str(correlation_h3k27ac), prop=dict(size=15), frameon=True, loc='upper left')
	at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
	ax.add_artist(at)
	plt.savefig(output_file_prefix + '_h3k27ac_correlation.png')
	plt.clf()
	
	# Plot tf sum correlation
	fig, ax = plt.subplots()
	ax.scatter(Y[1], Y_pred[1].squeeze())
	ax.set_title("tf_sum Correlation")
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	at = AnchoredText("PCC:" + str(correlation_tf_sum), prop=dict(size=15), frameon=True, loc='upper left')
	at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
	ax.add_artist(at)
	plt.savefig(output_file_prefix + '_tf_sum_correlation.png')
	plt.clf()
	
	return correlation_h3k27ac, correlation_tf_sum
	

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
	plot_scatterplot(history, 'Dense_h3k27ac_Spearman', 'val_Dense_h3k27ac_Spearman', 'SCC', 'epoch', 'Model performance h3k27ac (Spearman)', model_output_folder + 'Model_' + model_id + '_' + name + '_h3k27ac_spearman.png')
	plot_scatterplot(history, 'Dense_tf_sum_Spearman', 'val_Dense_tf_sum_Spearman', 'SCC', 'epoch', 'Model performance tf_sum (Spearman)', model_output_folder + 'Model_' + model_id + '_' + name + '_tf_sum_spearman.png')
	
	plot_scatterplot(history, 'Dense_h3k27ac_Pearson', 'val_Dense_h3k27ac_Pearson', 'PCC', 'epoch', 'Model performance h3k27ac (Pearson)', model_output_folder + 'Model_' + model_id + '_' + name + '_h3k27ac_pearson.png')
	plot_scatterplot(history, 'Dense_tf_sum_Pearson', 'val_Dense_tf_sum_Pearson', 'PCC', 'epoch', 'Model performance tf_sum (Pearson)', model_output_folder + 'Model_' + model_id + '_' + name + '_tf_sum_pearson.png')

	plot_scatterplot(history, 'Dense_h3k27ac_loss', 'val_Dense_h3k27ac_loss', 'loss', 'epoch', 'Model loss h3k27ac', model_output_folder + 'Model_' + model_id + '_' + name + '_h3k27ac_loss.png')		
	plot_scatterplot(history, 'Dense_tf_sum_loss', 'val_Dense_tf_sum_loss', 'loss', 'epoch', 'Model loss tf_sum', model_output_folder + 'Model_' + model_id + '_' + name + '_tf_sum_loss.png')
	 
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
		f.write("name\ttype\tmodel\treplicate\tpcc_train_h3k27ac\tpcc_train_tf_sum\tpcc_val_h3k27ac\tpcc_val_tf_sum\tpcc_test_h3k27ac\tpcc_test_tf_sum\n")
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
