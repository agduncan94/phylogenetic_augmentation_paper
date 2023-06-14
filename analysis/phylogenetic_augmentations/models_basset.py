# ####################################################################################################################
# models.py
#
# Class to train Keras models on Drosophila S2 and CHEF data
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
import h5py
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats
import os.path
import utils_basset as utils
from sklearn import metrics

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
tf.debugging.set_log_device_placement(False)
ALPHABET_SIZE = 4

# ====================================================================================================================
# Generator code for loading data
# ====================================================================================================================


def get_batch(fasta_obj, Measurements, tasks, indices, batch_size, use_homologs=False):
    """
    Creates a batch of the input and one-hot encodes the sequences
    """
    sequence_length = len(fasta_obj.fasta_dict[fasta_obj.fasta_names[0]][0])

    # One-hot encode a batch of sequences
    seqs = fasta_obj.one_hot_encode_batch(
        indices, sequence_length, use_homologs)
    X = np.nan_to_num(seqs)
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    # Retrieve batch of measurements
    Y_batch = Measurements.iloc[indices]

    # Create final output
    Y = Y_batch

    return X_reshaped, Y


def get_batch_hdf5(split_type, hdf5_file, seq_ids, Measurements, batch_size, indices, use_homologs=False):
    sequence_length = 600

    seqs = utils.one_hot_encode_batch_hdf5(
        split_type, hdf5_file, seq_ids, sequence_length, use_homologs)

    X = np.nan_to_num(seqs)
    X_batch = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    # Retrieve batch of measurements
    Y_batch = Measurements.iloc[indices]

    return X_batch, Y_batch


def data_gen_hdf5(split_type, hdf5_file, y_file, num_samples, batch_size, shuffle_epoch_end=True, use_homologs=False, order=False, filtered_indices=None):
    # Get keys from HDF5 file
    seq_ids = []
    with h5py.File(hdf5_file, "r") as f:
        for seq_id in f[split_type + '/sequences'].keys():
            seq_ids.append(seq_id)

    # Read measurement file
    Measurements = pd.read_table(y_file, header=None)

    # Sample the seq ids and measurements
    if (filtered_indices is not None):
        seq_ids_filtered = [seq_ids[i] for i in filtered_indices]
        Measurements = Measurements.iloc[filtered_indices]
        Measurements.reset_index(drop=True, inplace=True)
    else:
        seq_ids_filtered = seq_ids
    seq_ids = None

    # Create the batch indices
    if not order:
        indices = np.random.choice(
            list(range(num_samples)), num_samples, replace=False)
    else:
        indices = list(range(num_samples))

    ii = 0
    while True:
        yield get_batch_hdf5(split_type, hdf5_file, seq_ids_filtered[ii:ii + batch_size], Measurements, batch_size, indices[ii:ii + batch_size], use_homologs)
        ii += batch_size
        if ii >= num_samples:
            ii = 0
            if shuffle_epoch_end:
                if not order:
                    indices = np.random.choice(
                        list(range(num_samples)), num_samples, replace=False)
                else:
                    indices = list(range(num_samples))


def data_gen(fasta_file, y_file, homolog_folder, num_samples, batch_size, tasks, shuffle_epoch_end=True, use_homologs=False, order=False, filtered_indices=None):
    """
    Generator function for loading input data in batches
    """
    # Read FASTA file
    fasta_obj = utils.fasta(fasta_file)

    # Add homologs to FASTA data structure
    if use_homologs:
        print(homolog_folder)
        directory = os.fsencode(homolog_folder)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(filename)
            if filename.endswith(".fa"):
                fasta_obj.add_homolog_sequences(
                    os.path.join(homolog_folder, filename))

    # Read measurement file
    Measurements = pd.read_table(y_file, header=None)

    # Sample the FASTA structure and measurements
    if (filtered_indices is not None):
        fasta_obj.sample_fasta(filtered_indices)
        Measurements = Measurements.iloc[filtered_indices]
        Measurements.reset_index(drop=True, inplace=True)

    # Create the batch indices
    n_data = len(fasta_obj.fasta_names)
    if not order:
        indices = np.random.choice(
            list(range(num_samples)), num_samples, replace=False)
    else:
        indices = list(range(n_data))

    ii = 0
    while True:
        yield get_batch(fasta_obj, Measurements, tasks, indices[ii:ii + batch_size], batch_size, use_homologs)
        ii += batch_size
        if ii >= num_samples:
            ii = 0
            if shuffle_epoch_end:
                if not order:
                    indices = np.random.choice(
                        list(range(num_samples)), num_samples, replace=False)
                else:
                    indices = list(range(n_data))


# ====================================================================================================================
# Models encoders and training tasks
# ====================================================================================================================


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
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    return model

# ====================================================================================================================
# Train models
# ====================================================================================================================


def train(model, model_type, use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, batch_size=128):
    # Parameters for model training
    epochs = 8
    early_stop = 10
    fine_tune_epochs = 4
    batch_size = 512

    # Create a unique identifier for the model
    model_id = model_type + "_rep" + \
        str(replicate) + "_frac" + str(sample_fraction)

    # Create the output folder
    model_output_folder = output_folder + model_id + "/"
    os.makedirs(model_output_folder, exist_ok=True)

    # Determine the number of sequences in the train/val/test sets
    num_samples_train = utils.count_lines_in_file(
        file_folder + "Sequences_activity_Train.txt")
    num_samples_val = utils.count_lines_in_file(
        file_folder + "Sequences_activity_Val.txt")
    num_samples_test = utils.count_lines_in_file(
        file_folder + "Sequences_activity_Test.txt")

    # Sample a reduced set of sequences for training
    reduced_num_samples_train = int(num_samples_train * sample_fraction)
    filtered_indices = np.random.choice(
        list(range(num_samples_train)), reduced_num_samples_train, replace=False)

    # Data generators for train and val sets used during initial training
    datagen_train = data_gen_hdf5("training", file_folder + "augmentation_data.hdf5", file_folder + "Sequences_activity_Train.txt",
                                  reduced_num_samples_train, batch_size, use_homologs=use_homologs, filtered_indices=filtered_indices)

    datagen_val = data_gen_hdf5("validation", file_folder + "augmentation_data.hdf5", file_folder + "Sequences_activity_Val.txt",
                                num_samples_val, batch_size)

    # Fit model using the data generators
    history = model.fit(datagen_train,
                        validation_data=datagen_val,
                        epochs=epochs,
                        steps_per_epoch=math.floor(
                            reduced_num_samples_train / batch_size),
                        validation_steps=math.floor(
                            num_samples_val / batch_size),
                        callbacks=[EarlyStopping(patience=early_stop, monitor="val_loss", restore_best_weights=True),
                                   History()])

    print(history.history)
    # Save model (no finetuning)
    if use_homologs:
        augmentation_type = 'homologs'
    else:
        augmentation_type = 'none'

    save_model(model_id + "_" + augmentation_type,
               model, history, model_output_folder)

    # Write performance metrics to file (no finetuning)
    epochs = len(history.history['loss'])
    accuracy = history.history['accuracy'][epochs-1]
    validation_accuracy = history.history['val_accuracy'][epochs-1]
    auc = history.history['auc'][epochs-1]
    validation_auc = history.history['val_auc'][epochs-1]
    write_to_file(model_id, augmentation_type, model_type, replicate,
                  sample_fraction, history, tasks, accuracy, validation_accuracy, auc, validation_auc, output_folder)

    # Save plots for performance and loss (no finetuning)
    plot_scatterplots(history, model_output_folder,
                      model_id, augmentation_type)

    plot_prediction_vs_actual(
        model, file_folder + "augmentation_data.hdf5", file_folder + "Sequences_activity_Test.txt", model_output_folder + 'Model_' + model_id + "_" + augmentation_type + "_Test", num_samples_test, tasks, homolog_folder, False, batch_size)

    # Perform finetuning on the original training only
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
                  loss=['binary_crossentropy'],
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    # Update data generator to not use homologs (not needed for fine-tuning)
    if use_homologs:
        datagen_train = data_gen_hdf5("training", file_folder + "augmentation_data.hdf5", file_folder + "Sequences_activity_Train.txt",
                                      reduced_num_samples_train, batch_size, filtered_indices=filtered_indices)

    # Fit the model using new generator
    fine_tune_history = model.fit(datagen_train,
                                  validation_data=datagen_val,
                                  steps_per_epoch=math.floor(
                                      reduced_num_samples_train / batch_size),
                                  validation_steps=math.floor(
                                      num_samples_val / batch_size),
                                  epochs=fine_tune_epochs)

    # Save model (with finetuning)
    if use_homologs:
        augmentation_ft_type = 'homologs_finetune'
    else:
        augmentation_ft_type = 'finetune'

    save_model(model_id + "_" + augmentation_ft_type, model,
               fine_tune_history, model_output_folder)

    # Write performance metrics to file (with finetuning)
    epochs = len(fine_tune_history.history['loss'])
    accuracy = fine_tune_history.history['accuracy'][epochs-1]
    validation_accuracy = fine_tune_history.history['val_accuracy'][epochs-1]
    auc = fine_tune_history.history['auc_1'][epochs-1]
    validation_auc = fine_tune_history.history['val_auc_1'][epochs-1]
    write_to_file(model_id, augmentation_ft_type, model_type, replicate,
                  sample_fraction, fine_tune_history, tasks, accuracy, validation_accuracy, auc, validation_auc, output_folder)

    # Save plots for performance and loss (with finetuning)
    plot_scatterplots(fine_tune_history, model_output_folder,
                      model_id, augmentation_ft_type)

    plot_prediction_vs_actual(
        model, file_folder + "augmentation_data.hdf5", file_folder + "Sequences_activity_Test.txt", model_output_folder + 'Model_' + model_id + "_" + augmentation_ft_type + "_Test", num_samples_test, tasks, homolog_folder, False, batch_size)


def train_basset(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type="Basset", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = BassetEncoder(sequence_size)
    model = basset_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks)

# ====================================================================================================================
# Plot model performance for basset
# ====================================================================================================================


def plot_prediction_vs_actual(model, aug_file, activity_file, output_file_prefix, num_samples, homolog_dir, tasks, use_homologs=False, batch_size=128):
    # Load the activity data
    Y = pd.DataFrame()

    count = 0
    # for x, y in data_gen(fasta_file, activity_file, homolog_dir, num_samples, batch_size, tasks, use_homologs=use_homologs, order=True):
    for x, y in data_gen_hdf5("testing", aug_file, activity_file, num_samples, batch_size, use_homologs=use_homologs, order=True):
        Y = pd.concat((Y, y))
        count += 1
        if count > math.floor(num_samples / batch_size):
            break

    # Get model predictions
    data_generator = data_gen_hdf5("testing", aug_file, activity_file,
                                   num_samples, batch_size, use_homologs=use_homologs, order=True)
    Y_pred = model.predict(
        data_generator, steps=math.ceil(num_samples / batch_size))

    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
    for i in range(50):
        fpr, tpr, thresholds = metrics.roc_curve(
            Y.iloc[:, i], Y_pred[:, i])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' %
                  (str(i), metrics.auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')

    print(metrics.roc_auc_score(Y.to_numpy(), Y_pred, average=None))
    print(metrics.roc_auc_score(Y.to_numpy(), Y_pred))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.savefig(output_file_prefix + "_roc.png")


def plot_scatterplot(history, a, b, x, y, title, filename):
    """Plots a scatterplot and saves to file"""
    plt.plot(history.history[a])
    plt.plot(history.history[b])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename)
    plt.clf()


def plot_scatterplots(history, model_output_folder, model_id, name):
    """Plots model performance and loss for each task of a given model"""
    plot_scatterplot(history, 'accuracy', 'val_accuracy', 'epoch', 'accuracy', 'Model performance ' +
                     ' (accuracy)', model_output_folder + 'Model_' + model_id + '_' + name + '_accuracy.png')
    plot_scatterplot(history, 'loss', 'val_loss', 'epoch', 'loss',
                     'Model loss', model_output_folder + 'Model_' + model_id + '_' + name + '_loss.png')

# ====================================================================================================================
# Helpers
# ====================================================================================================================


def write_to_file(model_id, augmentation_type, model_type, replicate, sample_fraction, history, tasks, accuracy, validation_accuracy, auc, validation_auc, output_folder):
    """Writes model performance to a file"""

    correlation_file_path = output_folder + 'model_metrics.tsv'

    # Generate line to write to file
    line = model_id + "\t" + augmentation_type + "\t" + model_type + \
        "\t" + str(replicate) + "\t" + str(sample_fraction) + \
        "\t" + str(accuracy) + "\t" + str(validation_accuracy) + \
        "\t" + str(auc) + "\t" + str(validation_auc) + "\n"

    # Write line to file (and also header if necessary)
    if os.path.isfile(correlation_file_path):
        f = open(correlation_file_path, "a")
        f.write(line)
        f.close()
    else:
        f = open(correlation_file_path, "w")
        header_line = "name\ttype\tmodel\treplicate\tfraction\taccuracy\tval_accuracy\tauc\tval_auc\n"

        f.write(header_line)
        f.write(line)
        f.close()


def save_model(model_name, model, history, model_output_folder):
    """Saves a model and its history to a file"""
    model_json = model.to_json()
    with open(model_output_folder + 'Model_' + model_name + '.json', "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_output_folder + 'Model_' + model_name + '.h5')

    with open(model_output_folder + 'Model_' + model_name + '_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
