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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, auc, roc_curve, average_precision_score

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
tf.debugging.set_log_device_placement(False)
file_folder = "../process_data/output/classification_balanced/"
homolog_dir = "../process_data/output/classification_balanced/orthologs/"
output_folder = "./output_classification_balanced/"
correlation_file_path = output_folder + 'model_correlation.tsv'
batch_size = 32
fold = 4

# ====================================================================================================================
# Common model code
# ====================================================================================================================
def get_batch(fasta_obj, class_array, indices, batch_size, use_homologs=False, fold=1):
	"""
	Creates a batch of the input and one-hot encodes the sequences
	"""
	sequence_length = len(fasta_obj.fasta_dict[fasta_obj.fasta_names[0]][0])
	
	# One-hot encode the sequences
	seqs, seq_multiplier = fasta_obj.one_hot_encode_batch(indices, sequence_length, use_homologs, fold)
	X = np.nan_to_num(seqs)
	X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
	# Get batch of activity values
	class_array_batch = class_array[indices]
	
	if use_homologs:
		Y = np.repeat(class_array_batch.to_numpy(), seq_multiplier)
	else:
		Y = class_array_batch.to_numpy()
		
	return X, Y

def data_gen(fasta_file, class_file, homolog_dir, num_samples, batch_size, shuffle_epoch_end=True, use_homologs=False, fold=1, order=False):
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
	ClassFile = pd.read_table(class_file)
	class_array = ClassFile['class']
	
	# One hot encode
	#class_array = keras.utils.to_categorical(class_array, num_classes=2)
	
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
		
		yield get_batch(fasta_obj, class_array, indices[ii:ii + new_batch_size], new_batch_size, use_homologs, fold)
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
	
	# Classification head
    outputs = kl.Dense(1, activation='sigmoid', name=str('Dense_class'))(bottleneck)

    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

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
	
    # Classification head
    #outputs = (kl.Dense(1, activation='linear', name=str('Dense_linear'))(cnns))
    outputs = kl.Dense(1, activation='sigmoid', name=str('Dense_class'))(cnns)

    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=0.002),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

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
	
    # Classification head
    outputs = kl.Dense(1, activation='sigmoid', name=str('Dense_class'))(bottleneck)
    
    model = keras.models.Model([input], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=lr),
                  loss=['binary_crossentropy'],
                  metrics=['accuracy'])

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

	# Data for train and validation sets
	datagen_train = data_gen(file_folder + "Sequences_Train.fa", file_folder + "Sequences_activity_Train.txt", homolog_dir, num_samples_train, batch_size, True, use_homologs, fold)
	datagen_val = data_gen(file_folder + "Sequences_Val.fa", file_folder + "Sequences_activity_Val.txt", homolog_dir, num_samples_val, batch_size, True, False, fold)

	# Fit model
	history=model.fit(datagen_train,
                                  validation_data=datagen_val,
								  epochs=epochs,
								  steps_per_epoch=math.ceil(num_samples_train / batch_size),
								  validation_steps=math.ceil(num_samples_val / batch_size),
								  #class_weight={0: 1, 1: 15},
                                  callbacks=[EarlyStopping(patience=early_stop, monitor="val_loss", restore_best_weights=True),
                                             History()])
	
	# Save results
	if use_homologs:
		save_model(model_id + "_homologs", model, history, model_output_folder)
	else:
		save_model(model_id + "_none", model, history, model_output_folder)
		
	# Make a confusion matrix and precision recall curve (use scikit learn)
	if use_homologs:
		plot_confusion_matrix(model_id + "_homologs", model, file_folder + "Sequences_Test.fa", file_folder + "Sequences_activity_Test.txt", homolog_dir, num_samples_test)
	else:
		plot_confusion_matrix(model_id + "_none", model, file_folder + "Sequences_Test.fa", file_folder + "Sequences_activity_Test.txt", homolog_dir, num_samples_test)
	
    	
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
# Helpers
# ====================================================================================================================
def save_model(model_name, model, history, model_output_folder):
	# Save the model and history
	model_json = model.to_json()
	with open(model_output_folder + 'Model_' + model_name + '.json', "w") as json_file:
	    json_file.write(model_json)
		
	model.save_weights(model_output_folder + 'Model_' + model_name + '.h5')
	
	with open(model_output_folder + 'Model_' + model_name + '_history', 'wb') as file_pi:
	    pickle.dump(history.history, file_pi)

def plot_confusion_matrix(model_name, model, fasta_file, activity_file, homolog_dir, num_samples):
	# Load the activity data
	Y = np.array([])
	
	count = 0
	for x,y in data_gen(fasta_file, activity_file, homolog_dir, num_samples, batch_size, use_homologs=False, order=True):
		Y = np.concatenate((Y, y), axis=0)
		count += 1
		if count > (num_samples / batch_size):
			break
			
	# Get model predictions
	data_generator = data_gen(fasta_file, activity_file, homolog_dir, num_samples, batch_size, use_homologs=False, order=True)
	Y_pred = model.predict(data_generator, steps=math.ceil(num_samples / batch_size))
	
	Y_pred[Y_pred <= 0.5] = 0.
	Y_pred[Y_pred > 0.5] = 1.

	ConfusionMatrixDisplay.from_predictions(Y, Y_pred)
	plt.savefig(model_name + "_confusionmatrix.png")
	plt.clf()
	
	display = PrecisionRecallDisplay.from_predictions(Y, Y_pred, name="LinearSVC")
	_ = display.ax_.set_title("2-class Precision-Recall curve")
	plt.savefig(model_name + "_pr_curve.png")
	
	fpr, tpr, thresholds = roc_curve(Y, Y_pred)
	print(auc(fpr, tpr))
	
	print(average_precision_score(Y, Y_pred))
	
	