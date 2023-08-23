# ####################################################################################################################
# ml_models.py
#
# Class that contains model encoder and decoder implementations
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
from keras import backend as K
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
ALPHABET_SIZE = 4

# ====================================================================================================================
# Custom loss and metric functions
# ====================================================================================================================


def Pearson(y_true, y_pred):
    """
    Calculate the correlation between measured and predicted
    """
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
# Models encoders
# ====================================================================================================================


def DeepSTARREncoder(sequence_size):
    """Encoder for DeepSTARR from de Almeida et al"""

    params = {
        'kernel_size1': 7,
        'kernel_size2': 3,
        'kernel_size3': 5,
        'kernel_size4': 3,
        'num_filters': 256,
        'num_filters2': 60,
        'num_filters3': 60,
        'num_filters4': 120,
        'n_conv_layer': 4,
        'n_add_layer': 2,
        'dropout_prob': 0.4,
        'dense_neurons1': 256,
        'dense_neurons2': 256,
        'pad': 'same'
    }

    # Input shape
    input_shape = kl.Input(shape=(sequence_size, ALPHABET_SIZE))

    # Define encoder to create embedding vector
    x = kl.Conv1D(params['num_filters'], kernel_size=params['kernel_size1'],
                  padding=params['pad'],
                  name='Conv1D_1st')(input_shape)
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

    return input_shape, encoder


def ExplaiNNEncoder(sequence_size):
    """Encoder for ExplaiNN from Novakosky et al"""
    # Define parameters for the encoder
    params = {
        'padding': 'same',
        'conv1_kernel_size': 19,
        'fc_1_size': 20,
        'fc_2_size': 1,
        'num_of_motifs': 256,
        'dropout': 0.3
    }

    # Input shape
    input_shape = kl.Input(shape=(sequence_size, ALPHABET_SIZE))

    # Each CNN unit represents a motif
    encoder = []

    for i in range(params['num_of_motifs']):
        # 1st convolutional layer
        cnn_x = kl.Conv1D(1, kernel_size=params['conv1_kernel_size'], padding='same', name=str(
            'cnn_' + str(i)))(input_shape)
        cnn_x = BatchNormalization()(cnn_x)
        cnn_x = Activation('exponential')(cnn_x)
        cnn_x = MaxPooling1D(pool_size=7, strides=7)(cnn_x)
        cnn_x = Flatten()(cnn_x)

        # 1st FC layer
        cnn_x = kl.Dense(params['fc_1_size'], name=str(
            'FC_' + str(i) + '_a'))(cnn_x)
        cnn_x = BatchNormalization()(cnn_x)
        cnn_x = Activation('relu')(cnn_x)
        cnn_x = Dropout(params['dropout'])(cnn_x)

        # 2nd FC layer
        cnn_x = kl.Dense(params['fc_2_size'], name=str(
            'FC_' + str(i) + '_b'))(cnn_x)
        cnn_x = BatchNormalization()(cnn_x)
        cnn_x = Activation('relu')(cnn_x)
        cnn_x = Flatten()(cnn_x)

        encoder.append(cnn_x)

    encoder = kl.concatenate(encoder)

    return input_shape, encoder


def MotifDeepSTARREncoder(sequence_size):
    """Encoder for a model like DeepSTARR, but with an interpretable motif layer"""

    # Define parameters for the encoder
    params = {
        'padding': 'same',
        'conv1_kernel_size': 19,
        'conv1_shape': 256,
        'conv1_pool_size': 10,
        'dense_shape': 256,
        'dropout': 0.4
    }

    # Input shape
    input_shape = kl.Input(shape=(sequence_size, ALPHABET_SIZE))

    # Define encoder to create embedding vector
    encoder = kl.Conv1D(params['conv1_shape'], kernel_size=params['conv1_kernel_size'],
                        padding=params['padding'],
                        name='Conv1D')(input_shape)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('exponential')(encoder)
    encoder = MaxPooling1D(params['conv1_pool_size'])(encoder)
    encoder = Flatten()(encoder)

    # First dense layer
    encoder = kl.Dense(params['dense_shape'], name='Dense_a')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dropout(params['dropout'])(encoder)

    # Second dense layer
    encoder = kl.Dense(params['dense_shape'], name='Dense_b')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Dropout(params['dropout'])(encoder)

    return input_shape, encoder


def BassetEncoder(sequence_size):
    """Encoder for Basset from Kelley et al"""
    params = {
        'kernel_size1': 19,
        'kernel_size2': 11,
        'kernel_size3': 7,
        'num_filters1': 300,
        'num_filters2': 200,
        'num_filters3': 200,
        'max_pool1': 3,
        'max_pool2': 4,
        'max_pool3': 4,
        'n_add_layer': 2,
        'dropout_prob': 0.3,
        'dense_neurons1': 1000,
        'dense_neurons2': 1000,
        'pad': 'same'}

    # Input shape
    input_shape = kl.Input(shape=(sequence_size, ALPHABET_SIZE))

    # Define encoder to create embedding vector

    # First conv layer
    x = kl.Conv1D(params['num_filters1'], kernel_size=params['kernel_size1'],
                  padding=params['pad'],
                  name='Conv1D_1')(input_shape)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(params['max_pool1'])(x)

    # Second conv layer
    x = kl.Conv1D(params['num_filters2'], kernel_size=params['kernel_size2'],
                  padding=params['pad'],
                  name='Conv1D_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(params['max_pool2'])(x)

    # Third conv layer
    x = kl.Conv1D(params['num_filters3'], kernel_size=params['kernel_size3'],
                  padding=params['pad'],
                  name='Conv1D_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(params['max_pool3'])(x)

    x = Flatten()(x)

    # First linear layer
    x = kl.Dense(params['dense_neurons1'],
                 name=str('Dense_1'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(params['dropout_prob'])(x)

    # Second linear layer
    x = kl.Dense(params['dense_neurons2'],
                 name=str('Dense_2'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(params['dropout_prob'])(x)

    encoder = x

    return input_shape, encoder

# ====================================================================================================================
# Models decoders
# ====================================================================================================================


def n_regression_head(input_shape, encoder, tasks):
    """Regression head that supports an arbitrary number of tasks"""
    params = {
        'lr': 0.002
    }

    # Create prediction head per task
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear',
                       name=str('Dense_' + task))(encoder))

    model = keras.models.Model([input_shape], outputs)
    model.compile(keras.optimizers.Adam(learning_rate=params['lr']),
                  loss=['mse'] * len(tasks),
                  loss_weights=[1] * len(tasks),
                  metrics=[Pearson])

    return model


def basset_head(input_shape, encoder, tasks):
    """Basset head that supports 164 binary predictions"""
    params = {
        'lr': 0.002
    }

    # Create prediction head per task
    output = kl.Dense(
        len(tasks), activation='sigmoid', name=str('Dense_binary'))(encoder)

    model = keras.models.Model([input_shape], output)
    model.compile(keras.optimizers.Adam(learning_rate=params['lr']),
                  loss=['binary_crossentropy'],
                  metrics=[tf.keras.metrics.AUC(curve='PR', name="auc_pr"), tf.keras.metrics.AUC(name="auc_roc")])

    return model

# ====================================================================================================================
# Helpers
# ====================================================================================================================

def save_model(model_name, model, history, model_output_folder):
    """Saves a model and its history to a file"""
    model_json = model.to_json()
    with open(model_output_folder + 'Model_' + model_name + '.json', "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_output_folder + 'Model_' + model_name + '.h5')

    with open(model_output_folder + 'Model_' + model_name + '_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
