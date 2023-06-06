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
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats
import os.path
import utils_fold as utils

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
tf.debugging.set_log_device_placement(False)
ALPHABET_SIZE = 4

# ====================================================================================================================
# Generator code for loading data
# ====================================================================================================================


def get_batch(fasta_obj, Measurements, tasks, indices, batch_size, use_homologs=False, fold=1):
    """
    Creates a batch of the input and one-hot encodes the sequences
    """
    sequence_length = len(fasta_obj.fasta_dict[fasta_obj.fasta_names[0]][0])

    # One-hot encode a batch of sequences
    seqs, seq_multiplier = fasta_obj.one_hot_encode_batch(
        indices, sequence_length, use_homologs, fold)
    X = np.nan_to_num(seqs)
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    # Retrieve batch of measurements
    Y_batch = []
    for i, task in enumerate(tasks):
        Y_batch.append(Measurements[Measurements.columns[i]][indices])

    # Create final output
    if use_homologs:
        Y = [np.repeat(item.to_numpy(), seq_multiplier) for item in Y_batch]
    else:
        Y = [item.to_numpy() for item in Y_batch]

    return X_reshaped, Y


def data_gen(fasta_file, y_file, homolog_folder, num_samples, batch_size, tasks, shuffle_epoch_end=True, use_homologs=False, fold=1, order=False, filtered_indices=None):
    """
    Generator function for loading input data in batches
    """
    # Read FASTA file
    fasta_obj = utils.fasta(fasta_file)

    # Add homologs to FASTA data structure
    if use_homologs:
        directory = os.fsencode(homolog_folder)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".fa"):
                fasta_obj.add_homolog_sequences(
                    os.path.join(homolog_folder, filename))

    # Read measurement file
    Measurements = pd.read_table(y_file)

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
        yield get_batch(fasta_obj, Measurements, tasks, indices[ii:ii + batch_size], batch_size, use_homologs, fold)
        ii += batch_size
        if ii >= num_samples:
            ii = 0
            if shuffle_epoch_end:
                if not order:
                    indices = np.random.choice(
                        list(range(num_samples)), num_samples, replace=False)
                else:
                    indices = list(range(n_data))


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
# Models encoders and training tasks
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
        'num_of_motifs': 100,
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
        'conv1_shape': 128,
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
    encoder = Activation('relu')(encoder)
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


# ====================================================================================================================
# Train models
# ====================================================================================================================


def train(model, model_type, use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, batch_size=128):
    # Parameters for model training
    epochs = 100
    early_stop = 10
    fine_tune_epochs = 10
    fold = 4

    # Create a unique identifier for the model
    model_id = model_type + "_rep" + \
        str(replicate) + "_frac" + str(sample_fraction)

    # Create the output folder
    model_output_folder = output_folder + model_id + "/"
    os.makedirs(model_output_folder, exist_ok=True)

    # Determine the number of sequences in the train/val/test sets
    num_samples_train = utils.count_lines_in_file(
        file_folder + "Sequences_activity_Train.txt") - 1
    num_samples_val = utils.count_lines_in_file(
        file_folder + "Sequences_activity_Val.txt") - 1
    num_samples_test = utils.count_lines_in_file(
        file_folder + "Sequences_activity_Test.txt") - 1

    # Sample a reduced set of sequences for training
    reduced_num_samples_train = int(num_samples_train * sample_fraction)
    filtered_indices = np.random.choice(
        list(range(num_samples_train)), reduced_num_samples_train, replace=False)

    # Data generators for train and val sets used during initial training
    datagen_train = data_gen(file_folder + "Sequences_Train.fa", file_folder + "Sequences_activity_Train.txt",
                             homolog_folder, reduced_num_samples_train, batch_size, tasks, True, use_homologs, fold, False, filtered_indices)
    datagen_val = data_gen(file_folder + "Sequences_Val.fa", file_folder + "Sequences_activity_Val.txt",
                           homolog_folder, num_samples_val, batch_size, tasks, True, False, fold, False, None)

    # Fit model using the data generators
    history = model.fit(datagen_train,
                        validation_data=datagen_val,
                        epochs=epochs,
                        steps_per_epoch=math.ceil(
                            reduced_num_samples_train / batch_size),
                        validation_steps=math.ceil(
                            num_samples_val / batch_size),
                        callbacks=[EarlyStopping(patience=early_stop, monitor="val_loss", restore_best_weights=True),
                                   History()])

    # Save model (no finetuning)
    if use_homologs:
        augmentation_type = 'homologs'
    else:
        augmentation_type = 'none'

    save_model(model_id + "_" + augmentation_type,
               model, history, model_output_folder)

    # Plot test performance on a scatterplot (no finetuning)
    test_correlations = plot_prediction_vs_actual(model, file_folder + "Sequences_Test.fa",
                                                  file_folder + "Sequences_activity_Test.txt",
                                                  model_output_folder + 'Model_' + model_id + "_" + augmentation_type + "_Test",
                                                  num_samples_test,
                                                  homolog_folder,
                                                  tasks,
                                                  False,
                                                  batch_size)

    # Write performance metrics to file (no finetuning)
    write_to_file(model_id, augmentation_type, model_type, replicate,
                  sample_fraction, history, tasks, test_correlations, output_folder)

    # Save plots for performance and loss (no finetuning)
    plot_scatterplots(history, model_output_folder,
                      model_id, augmentation_type, tasks)

    # Perform finetuning on the original training only
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
                  loss=['mse'] * len(tasks),
                  loss_weights=[1] * len(tasks),
                  metrics=[Pearson])

    # Update data generator to not use homologs (not needed for fine-tuning)
    if use_homologs:
        datagen_train = data_gen(file_folder + "Sequences_Train.fa", file_folder + "Sequences_activity_Train.txt",
                                 homolog_folder, reduced_num_samples_train, batch_size, tasks, True, False, fold, False, filtered_indices)

    # Fit the model using new generator
    fine_tune_history = model.fit(datagen_train,
                                  validation_data=datagen_val,
                                  steps_per_epoch=math.ceil(
                                      reduced_num_samples_train / batch_size),
                                  validation_steps=math.ceil(
                                      num_samples_val / batch_size),
                                  epochs=fine_tune_epochs)

    # Save model (with finetuning)
    if use_homologs:
        augmentation_ft_type = 'homologs_finetune'
    else:
        augmentation_ft_type = 'finetune'

    save_model(model_id + "_" + augmentation_ft_type, model,
               fine_tune_history, model_output_folder)

    # Plot test performance on a scatterplot (with finetuning)
    test_correlations = plot_prediction_vs_actual(model, file_folder + "Sequences_Test.fa",
                                                  file_folder + "Sequences_activity_Test.txt",
                                                  model_output_folder + 'Model_' + model_id +
                                                  "_" + augmentation_ft_type + "_Test",
                                                  num_samples_test,
                                                  homolog_folder,
                                                  tasks,
                                                  False,
                                                  batch_size)

    # Write performance metrics to file (with finetuning)
    write_to_file(model_id, augmentation_ft_type, model_type, replicate,
                  sample_fraction, fine_tune_history, tasks, test_correlations, output_folder)

    # Save plots for performance and loss (with finetuning)
    plot_scatterplots(fine_tune_history, model_output_folder,
                      model_id, augmentation_ft_type, tasks)


def train_deepstarr(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type="DeepSTARR", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = DeepSTARREncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks)


def train_explainn(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type="ExplaiNN", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = ExplaiNNEncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks)


def train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, model_type="MotifDeepSTARR", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = MotifDeepSTARREncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks)

# ====================================================================================================================
# Plot model performance for dual regression
# ====================================================================================================================


def plot_prediction_vs_actual(model, fasta_file, activity_file, output_file_prefix, num_samples, homolog_folder, tasks, use_homologs=False, batch_size=128):
    """Plots the predicted vs actual activity for each task on given input set"""

    # Load the activity data
    Y = []
    for task in tasks:
        Y.append(np.array([]))

    count = 0
    for x, y in data_gen(fasta_file, activity_file, homolog_folder, num_samples, batch_size, tasks, use_homologs=use_homologs, order=True):
        for i, task in enumerate(tasks):
            Y[i] = np.concatenate((Y[i], y[i]), axis=0)
        count += 1
        if count > (num_samples / batch_size):
            break

    # Get model predictions
    data_generator = data_gen(fasta_file, activity_file, homolog_folder,
                              num_samples, batch_size, tasks, use_homologs=use_homologs, order=True)
    Y_pred = model.predict(
        data_generator, steps=math.ceil(num_samples / batch_size))

    correlations = []
    # Make plots for each task
    for i, task in enumerate(tasks):
        correlation_y = stats.pearsonr(Y[i], Y_pred[i].squeeze())[0]

        fig, ax = plt.subplots()
        ax.scatter(Y[i], Y_pred[i].squeeze())
        ax.set_title(task + " Correlation")
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        at = AnchoredText("PCC:" + str(correlation_y),
                          prop=dict(size=15), frameon=True, loc='upper left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        plt.savefig(output_file_prefix + '_' + task + '_correlation.png')
        plt.clf()
        correlations.append(correlation_y)

    # Return a list of correlations for all tasks
    return correlations


def plot_scatterplot(history, a, b, x, y, title, filename):
    """Plots a scatterplot and saves to file"""
    plt.plot(history.history[a])
    plt.plot(history.history[b])
    plt.title(title)
    plt.ylabel(x)
    plt.xlabel(y)
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename)
    plt.clf()


def plot_scatterplots(history, model_output_folder, model_id, name, tasks):
    """Plots model performance and loss for each task of a given model"""
    for task in tasks:
        plot_scatterplot(history, 'Dense_' + task + '_Pearson', 'val_Dense_' + task + '_Pearson', 'PCC', 'epoch', 'Model performance ' +
                         task + ' (Pearson)', model_output_folder + 'Model_' + model_id + '_' + name + '_' + task + '_pearson.png')
        plot_scatterplot(history, 'Dense_' + task + '_loss', 'val_Dense_' + task + '_loss', 'loss', 'epoch',
                         'Model loss ' + task, model_output_folder + 'Model_' + model_id + '_' + name + '_' + task + '_loss.png')

# ====================================================================================================================
# Helpers
# ====================================================================================================================


def write_to_file(model_id, augmentation_type, model_type, replicate, sample_fraction, history, tasks, test_correlations, output_folder):
    """Writes model performance to a file"""

    correlation_file_path = output_folder + 'model_correlation.tsv'

    # Generate line to write to file
    line = model_id + "\t" + augmentation_type + "\t" + model_type + \
        "\t" + str(replicate) + "\t" + str(sample_fraction) + "\t"

    epochs_total = len(history.history['val_Dense_' + tasks[0] + '_Pearson'])
    for i, task in enumerate(tasks):
        line += str(history.history['Dense_' +
                    task + '_Pearson'][epochs_total-1]) + "\t"
    for i, task in enumerate(tasks):
        line += str(history.history['val_Dense_' +
                    task + '_Pearson'][epochs_total-1]) + "\t"
    for i, task in enumerate(tasks):
        if i == len(tasks) - 1:
            line += str(test_correlations[i]) + "\n"
        else:
            line += str(test_correlations[i]) + "\t"

    # Write line to file (and also header if necessary)
    if os.path.isfile(correlation_file_path):
        f = open(correlation_file_path, "a")
        f.write(line)
        f.close()
    else:
        f = open(correlation_file_path, "w")
        header_line = "name\ttype\tmodel\treplicate\tfraction\t"
        for i, task in enumerate(tasks):
            header_line += "pcc_train_" + task + "\t"
        for i, task in enumerate(tasks):
            header_line += "pcc_val_" + task + "\t"
        for i, task in enumerate(tasks):
            if i == len(tasks) - 1:
                header_line += "pcc_test_" + task + "\n"
            else:
                header_line += "pcc_test_" + task + "\t"

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
