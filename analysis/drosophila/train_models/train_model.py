import tensorflow as tf
import tensorflow_addons as tfa
import keras
import keras.layers as kl
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers import BatchNormalization, InputLayer, Input
from keras import models
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
import pickle
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import os
import math
import random
import utils
random.seed(1234)

tf.debugging.set_log_device_placement(False)

batch_size = 128

# Define generator for creating batches
def get_batch(fasta_obj, dev_activity_array, hk_activity_array, indices, batch_size, use_homologs=False, fold=1):
	sequence_length = len(fasta_obj.fasta_dict[fasta_obj.fasta_names[0]][0])
	
	# One-hot encode the sequences
	seqs, seq_multiplier = fasta_obj.one_hot_encode_batch(indices, sequence_length, use_homologs, fold)
	X = np.nan_to_num(seqs)
	X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
	
	# Get batch of activity values
	dev_activity_array_batch = dev_activity_array[indices]
	hk_activity_array_batch = hk_activity_array[indices]
	
	if use_homologs:
		Y = [np.repeat(dev_activity_array_batch.to_numpy(), seq_multiplier), np.repeat(hk_activity_array_batch.to_numpy(), seq_multiplier)]
	else:
		Y = [dev_activity_array_batch.to_numpy(), hk_activity_array_batch.to_numpy()]
	
	return X_reshaped, Y
		
def data_gen(fasta_file, activity_file, homolog_dir, num_samples, batch_size, shuffle_epoch_end=True, use_homologs=False, fold=1):
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
	
	dev_activity_array = Activity['Dev_log2_enrichment']
	hk_activity_array = Activity['Hk_log2_enrichment']
	
	# Create the batch indices
	indices = np.random.choice(list(range(num_samples)), num_samples, replace=False)
	
	ii = 0
	while True:		
		# Calculate how large a batch size to make
		if use_homologs:
			new_batch_size = calculate_batch_size(fasta_obj, indices, batch_size, ii, fold)
		else:
			new_batch_size = batch_size
		
		yield get_batch(fasta_obj, dev_activity_array, hk_activity_array, indices[ii:ii + new_batch_size], new_batch_size, use_homologs, fold)
		ii += new_batch_size
		if ii >= num_samples:
			ii = 0
			if shuffle_epoch_end:
				indices = np.random.choice(list(range(num_samples)), num_samples, replace=False)
			

def calculate_batch_size(fasta_obj, indices, batch_size, ii, fold):
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
     return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), 
                       tf.cast(y_true, tf.float32)], Tout = tf.float32))

deepstarr_params = {'epochs': 100,
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

# Cite
def DeepSTARR(params=deepstarr_params):
    lr = params['lr']
    dropout_prob = params['dropout_prob']
    n_conv_layer = params['n_conv_layer']
    n_add_layer = params['n_add_layer']
    
    input = kl.Input(shape=(249, 4))
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
    
    # heads per task (Dev and Hk enhancer activities)
    tasks = ['Dev', 'Hk']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))

    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=lr),
                  loss=['mse', 'mse'], # loss
                  loss_weights=[1, 1], # loss weigths to balance
                  metrics=[Spearman]) # additional track metric

    return model, params

simple_cnn_params = {'epochs': 100,
          'early_stop': 10,
          'kernel_size': 19,
          'lr': 0.002,
          'num_filters': 256,
          'pad':'same'}

def simple_model(params=simple_cnn_params):
    lr = params['lr']
	
    input = kl.Input(shape=(249, 4))
    x = kl.Conv1D(params['num_filters'], kernel_size=params['kernel_size'],
                  padding=params['pad'],
                  name='Conv1D')(input)
    x = BatchNormalization()(x)
    x = Activation('exponential')(x)
    x = MaxPooling1D(2)(x)
    
    x = Flatten()(x)
    
    bottleneck = x
    
    # heads per task (Dev and Hk enhancer activities)
    tasks = ['Dev', 'Hk']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))

    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=lr),
                  loss=['mse', 'mse'], # loss
                  loss_weights=[1, 1], # loss weigths to balance
                  metrics=[Spearman]) # additional track metric

    return model, params


explainn_params = {'epochs': 100,
          'early_stop': 10,
          'kernel_size': 7,
          'lr': 0.002,
          'num_filters': 256,
          'dense_neurons1': 256,
          'dense_neurons2': 256,
          'pad':'same',
		  'num_cnns': 128}

def simple_model(params=explainn_params):
    lr = params['lr']
	
    input = kl.Input(shape=(249, 4))
    cnns = []
	
    for i in range(params['num_cnns']):
		# 1st convolutional layer
        cnn_x = kl.Conv1d(1, kernel_size=19, padding='same', name=str('cnn_' + str(i)))(input)
        cnn_x = BatchNormalization(cnn_x)
        cnn_x = Activation('exponential')(cnn_x)
        cnn_x = MaxPooling1D(pool_size=7, strides=7)(cnn_x)
        cnn_x = Flatten(cnn_x)
		
		# 1st FC layer
        cnn_x = kl.Dense(100, name=str('FC_' + str(i) + '_a'))(cnn_x) # What is the size?
        cnn_x = BatchNormalization(cnn_x)
        cnn_x = Activation('relu')(cnn_x)
        cnn_x = Dropout(0.3)(cnn_x)
		
		# 2nd FC layer
        cnn_x = kl.Dense(1, name=str('FC_' + str(i) + '_b'))(cnn_x) # What is the size?
        cnn_x = BatchNormalization(cnn_x)
        cnn_x = Activation('relu')(cnn_x)
        cnn_x = Flatten(cnn_x)
		
        cnns.append(cnn_x)
            
    # heads per task (Dev and Hk enhancer activities)
    tasks = ['Dev', 'Hk']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(cnns))

    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=lr),
                  loss=['mse', 'mse'], # loss
                  loss_weights=[1, 1], # loss weigths to balance
                  metrics=[Spearman]) # additional track metric

    return model, params

def train(model, datagen_train, datagen_val, num_samples_train, num_samples_val):
	params = {'epochs': 100, 'early_stop': 10}
	history=model.fit(datagen_train,
                                  validation_data=datagen_val,
								  epochs=params['epochs'],
								  steps_per_epoch=math.ceil(num_samples_train / batch_size),
								  validation_steps=math.ceil(num_samples_val / batch_size),
                                  callbacks=[EarlyStopping(patience=params['early_stop'], monitor="val_loss", restore_best_weights=True),
                                             History()])
    
	return model, history

def train_cnn(file_folder, homolog_dir, parent_folder, model_base_name, model_name, use_homologs, fold=4):
	# Output folder
	output_folder = parent_folder + '/' + model_base_name + '/'
	
	# Determine the number of sequences in training and validation
	num_samples_train = utils.count_lines_in_file(file_folder + "Sequences_activity_Train.txt") - 1
	num_samples_val = utils.count_lines_in_file(file_folder + "Sequences_activity_Val.txt") - 1
		
	print("Training " + model_name + " with num train: " + str(num_samples_train) + " and num val " + str(num_samples_val) + " and fold " + str(fold))
	
	# Data for train and validation sets
	datagen_train = data_gen(file_folder + "Sequences_Train.fa", file_folder + "Sequences_activity_Train.txt", homolog_dir, num_samples_train, batch_size, True, use_homologs, fold)
	datagen_val = data_gen(file_folder + "Sequences_Val.fa", file_folder + "Sequences_activity_Val.txt", homolog_dir, num_samples_val, batch_size, True, False, fold)

	# Train model
	main_model, main_params = DeepSTARR()
	main_model, history = train(main_model, datagen_train, datagen_val, num_samples_train, num_samples_val)
	
	# Do not use homologs for fine tuning
	if use_homologs:
		# Fine tune (https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW)
		main_model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),  # Very low learning rate
	 	              loss=['mse', 'mse'], # loss
	 	 			  loss_weights=[1, 1],
	 	              metrics=[Spearman])
		
		datagen_train = data_gen(file_folder + "Sequences_Train.fa", file_folder + "Sequences_activity_Train.txt", homolog_dir, num_samples_train, batch_size, True, False, fold)
	
		# Train end-to-end. Be careful to stop before you overfit!
		fine_tune_history = main_model.fit(datagen_train,
	                                   validation_data=datagen_val,
	 								  steps_per_epoch=math.ceil(num_samples_train / batch_size),
	 								  validation_steps=math.ceil(num_samples_val / batch_size),
	 								  epochs=10)
		with open(parent_folder + '/validation_scc.tsv', 'a') as file:
			dev_scc_val = fine_tune_history.history['val_Dense_Dev_Spearman'][9]
			hk_scc_val = fine_tune_history.history['val_Dense_Hk_Spearman'][9]
			dev_scc_train = history.history['Dense_Dev_Spearman'][9]
			hk_scc_train = history.history['Dense_Hk_Spearman'][9]
			file.write(str(dev_scc_train) + '\t' + str(hk_scc_train) + '\t' + str(dev_scc_val) + '\t' + str(hk_scc_val) + '\t10\thomologs_finetune\n')
			
	# Save the model
	model_json = main_model.to_json()
	with open(output_folder + 'Model_' + model_name + '.json', "w") as json_file:
	    json_file.write(model_json)
	main_model.save_weights(output_folder + 'Model_' + model_name + '.h5')
	
	with open(output_folder + 'Model_' + model_name + '_history', 'wb') as file_pi:
	    pickle.dump(history.history, file_pi)
			
	with open(parent_folder + '/validation_scc.tsv', 'a') as file:
		   epochs = len(history.history['val_Dense_Dev_Spearman'])
		   dev_scc_val = history.history['val_Dense_Dev_Spearman'][epochs-1]
		   hk_scc_val = history.history['val_Dense_Hk_Spearman'][epochs-1]
		   dev_scc_train = history.history['Dense_Dev_Spearman'][epochs-1]
		   hk_scc_train = history.history['Dense_Hk_Spearman'][epochs-1]
		   if use_homologs:
			      file.write(str(dev_scc_train) + '\t' + str(hk_scc_train) + '\t' + str(dev_scc_val) + '\t' + str(hk_scc_val) + '\t' + str(epochs) + '\thomologs\n')
		   else:
			      file.write(str(dev_scc_train) + '\t' + str(hk_scc_train) + '\t' + str(dev_scc_val) + '\t' + str(hk_scc_val) + '\t' + str(epochs) + '\tnone\n')
	
	# Make plots
	plt.plot(history.history['Dense_Hk_Spearman'])
	plt.plot(history.history['val_Dense_Hk_Spearman'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(output_folder + 'Model_' + model_name + '_hk_acc.png')
	plt.clf()
	
	plt.plot(history.history['Dense_Dev_Spearman'])
	plt.plot(history.history['val_Dense_Dev_Spearman'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(output_folder + 'Model_' + model_name + '_dev_acc.png')
	plt.clf()
	
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(output_folder + 'Model_' + model_name + '_loss.png')

