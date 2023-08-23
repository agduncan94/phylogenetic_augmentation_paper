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
import utils
from ml_models import *

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
tf.debugging.set_log_device_placement(False)
ALPHABET_SIZE = 4

# ====================================================================================================================
# Generator code for loading data
# ====================================================================================================================


def get_batch(fasta_obj, data, tasks, indices, batch_size, use_homologs=False, homolog_rate=1.0):
    """
    Creates a batch of the input and one-hot encodes the sequences
    """
    sequence_length = len(fasta_obj.fasta_dict[fasta_obj.fasta_names[0]][0])

    # One-hot encode a batch of sequences
    seqs = fasta_obj.one_hot_encode_batch(
        indices, sequence_length, use_homologs, homolog_rate)
    X = np.nan_to_num(seqs)
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    # Retrieve batch of measurements
    Y_batch = []
    for i, task in enumerate(tasks):
        Y_batch.append(data[data.columns[i]][indices])

    # Create final output
    Y = [item.to_numpy() for item in Y_batch]

    return X_reshaped, Y


def data_gen(input_file, homolog_folder, num_samples, batch_size, tasks, shuffle_epoch_end=True, use_homologs=False, species=None, sequence_filter=None, order=False, filtered_indices=None, homolog_rate=1.0):
    """
    Generator function for loading input data in batches
    """

    # Read input file
    data = pd.read_table(input_file)

    # Remove sequences not in list
    if sequence_filter is not None:
        data = data[data['Name'].str.contains('|'.join(sequence_filter))]
        data.reset_index(drop=True, inplace=True)

    # Sample the input data
    if (filtered_indices is not None):
        data = data.iloc[filtered_indices]
        data.reset_index(drop=True, inplace=True)

    # Create FASTA object with homologs
    fasta_obj = utils.fasta(data)

    # Add homologs to FASTA data structure
    if use_homologs:
        directory = os.fsencode(homolog_folder)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".fa") and (species is None or [ele for ele in species if(ele in filename)]):
                # print(filename)
                fasta_obj.add_homolog_sequences(
                    os.path.join(homolog_folder, filename))

    # Create the batch indices
    n_data = len(fasta_obj.fasta_names)
    if not order:
        indices = np.random.choice(
            list(range(num_samples)), num_samples, replace=False)
    else:
        indices = list(range(n_data))

    ii = 0
    while True:
        yield get_batch(fasta_obj, data, tasks, indices[ii:ii + batch_size], batch_size, use_homologs, homolog_rate)
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
# Train models
# ====================================================================================================================


def train(model, model_type, use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, homolog_rate=1.0, species=None, sequence_filter=None, batch_size=128):
    # Parameters for model training
    epochs = 100
    early_stop = 10
    fine_tune_epochs = 5
    batch_size = 256
    #baseline_filters = ['positive_peaks', 'negative']
    baseline_filters = None

    # Create a unique identifier for the model
    model_id = model_type + "_rep" + \
        str(replicate) + "_frac" + str(sample_fraction)

    # Create the output folder
    model_output_folder = output_folder + model_id + "/"
    os.makedirs(model_output_folder, exist_ok=True)

    # Determine the number of sequences in the train/val/test sets. Optionally filter based on seq name including a filter.
    train_file = file_folder + "Sequences_Train.txt"
    val_file = file_folder + "Sequences_Val.txt"
    test_file = file_folder + "Sequences_Test.txt"

    num_samples_train = utils.count_lines_in_file_with_filter(
        train_file, sequence_filter) - 1
    num_samples_val = utils.count_lines_in_file_with_filter(
        val_file, baseline_filters) - 1
    num_samples_test = utils.count_lines_in_file_with_filter(
        test_file, baseline_filters) - 1

    print('filtered training size: ' + str(num_samples_train))
    print('filtered val size: ' + str(num_samples_val))
    print('filtered test size: ' + str(num_samples_test))

    # Sample a reduced set of sequences for training
    reduced_num_samples_train = int(num_samples_train * sample_fraction)
    filtered_indices = np.random.choice(
        list(range(num_samples_train)), reduced_num_samples_train, replace=False)

    # Data generators for train and val sets used during initial training
    datagen_train = data_gen(train_file,
                             homolog_folder, reduced_num_samples_train, batch_size, tasks, True, use_homologs, species, sequence_filter, False, filtered_indices, homolog_rate)
    datagen_val = data_gen(val_file,
                           homolog_folder, num_samples_val, batch_size, tasks, True, False, None, baseline_filters, False, None, homolog_rate)

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
    # Define augmentation type
    if use_homologs:
        homolog_augmentation_type = 'homologs'
    else:
        homolog_augmentation_type = 'none'

    augmentation_type = 'none'
    if sequence_filter is not None:
        augmentation_type = '-'.join(sequence_filter)

    # Save model (no finetuning)
    save_model(model_id + "_" + homolog_augmentation_type + "_" + augmentation_type,
               model, history, model_output_folder)

    # Plot test performance on a scatterplot (no finetuning)
    test_correlations = plot_prediction_vs_actual(model, test_file,
                                                  model_output_folder + 'Model_' + model_id + "_" +
                                                  homolog_augmentation_type + "_" + augmentation_type + "_Test",
                                                  num_samples_test,
                                                  homolog_folder,
                                                  tasks,
                                                  False,
                                                  baseline_filters,
                                                  batch_size)

    # Write performance metrics to file (no finetuning)
    write_to_file(model_id, homolog_augmentation_type, augmentation_type, model_type, replicate,
                  sample_fraction, history, tasks, test_correlations, homolog_rate, output_folder)

    # Save plots for performance and loss (no finetuning)
    plot_scatterplots(history, model_output_folder,
                      model_id, homolog_augmentation_type, augmentation_type, tasks)

    # Perform finetuning on the original training only
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
                  loss=['mse'] * len(tasks),
                  loss_weights=[1] * len(tasks),
                  metrics=[Pearson])

    # Update data generator to not use homologs (not needed for fine-tuning)
    if use_homologs:
        datagen_train = data_gen(train_file,
                                 homolog_folder, reduced_num_samples_train, batch_size, tasks, True, False, None, sequence_filter, False, filtered_indices, homolog_rate)

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
        homolog_augmentation_ft_type = 'homologs_finetune'
    else:
        homolog_augmentation_ft_type = 'finetune'

    augmentation_ft_type = 'none'
    if sequence_filter is not None:
        augmentation_ft_type = '-'.join(sequence_filter)

    save_model(model_id + "_" + homolog_augmentation_ft_type + "_" + augmentation_ft_type, model,
               fine_tune_history, model_output_folder)

    # Plot test performance on a scatterplot (with finetuning)
    test_correlations = plot_prediction_vs_actual(model, test_file,
                                                  model_output_folder + 'Model_' + model_id +
                                                  "_" + homolog_augmentation_ft_type + "_" + augmentation_ft_type + "_Test",
                                                  num_samples_test,
                                                  homolog_folder,
                                                  tasks,
                                                  False,
                                                  baseline_filters,
                                                  batch_size)

    # Write performance metrics to file (with finetuning)
    write_to_file(model_id, homolog_augmentation_ft_type, augmentation_ft_type, model_type, replicate,
                  sample_fraction, fine_tune_history, tasks, test_correlations, homolog_rate, output_folder)

    # Save plots for performance and loss (with finetuning)
    plot_scatterplots(fine_tune_history, model_output_folder,
                      model_id, homolog_augmentation_ft_type, augmentation_ft_type, tasks)

    # Clean up
    del(model)
    keras.backend.clear_session()


def train_deepstarr(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, homolog_rate=1.0, species=None, sequence_filter=None, model_type="DeepSTARR", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = DeepSTARREncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks, homolog_rate, species, sequence_filter)


def train_explainn(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, homolog_rate=1.0, species=None, sequence_filter=None, model_type="ExplaiNN", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = ExplaiNNEncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks, homolog_rate, species, sequence_filter)


def train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, homolog_rate=1.0, species=None, sequence_filter=None, model_type="MotifDeepSTARR", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = MotifDeepSTARREncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks, homolog_rate, species, sequence_filter)


def train_motif_linear(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, homolog_rate=1.0, species=None, sequence_filter=None, model_type="MotifLinear", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = MotifLinearEncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks, homolog_rate, species, sequence_filter)


def train_motif_linear_relu(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, homolog_rate=1.0, species=None, sequence_filter=None, model_type="MotifLinearRelu", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = MotifLinearReluEncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks, homolog_rate, species, sequence_filter)


def train_basset(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, tasks, sequence_size, homolog_rate=1.0, species=None, sequence_filter=None, model_type="Basset", gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    input_shape, encoder = BassetEncoder(sequence_size)
    model = n_regression_head(input_shape, encoder, tasks)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, tasks, homolog_rate, species, sequence_filter)

# ====================================================================================================================
# Plot model performance for dual regression
# ====================================================================================================================


def plot_prediction_vs_actual(model, input_file, output_file_prefix, num_samples, homolog_folder, tasks, use_homologs=False, baseline_filters=None, batch_size=128):
    """Plots the predicted vs actual activity for each task on given input set"""

    # Load the activity data
    Y = []
    for task in tasks:
        Y.append(np.array([]))

    count = 0
    for x, y in data_gen(input_file, homolog_folder, num_samples, batch_size, tasks, use_homologs=use_homologs, sequence_filter=baseline_filters, order=True):
        for i, task in enumerate(tasks):
            Y[i] = np.concatenate((Y[i], y[i]), axis=0)
        count += 1
        if count > (num_samples / batch_size):
            break

    # Get model predictions
    data_generator = data_gen(input_file, homolog_folder,
                              num_samples, batch_size, tasks, use_homologs=use_homologs, sequence_filter=baseline_filters, order=True)
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


def plot_scatterplots(history, model_output_folder, model_id, name1, name2, tasks):
    """Plots model performance and loss for each task of a given model"""
    for task in tasks:
        plot_scatterplot(history, 'Dense_' + task + '_Pearson', 'val_Dense_' + task + '_Pearson', 'PCC', 'epoch', 'Model performance ' +
                         task + ' (Pearson)', model_output_folder + 'Model_' + model_id + '_' + name1 + '_' + name2 + '_' + task + '_pearson.png')
        plot_scatterplot(history, 'Dense_' + task + '_loss', 'val_Dense_' + task + '_loss', 'loss', 'epoch',
                         'Model loss ' + task, model_output_folder + 'Model_' + model_id + '_' + name1 + '_' + name2 + '_' + task + '_loss.png')

# ====================================================================================================================
# Helpers
# ====================================================================================================================


def write_to_file(model_id, homolog_augmentation_type, augmentation_type, model_type, replicate, sample_fraction, history, tasks, test_correlations, homolog_rate, output_folder):
    """Writes model performance to a file"""

    correlation_file_path = output_folder + 'model_correlation.tsv'

    # Generate line to write to file
    line = model_id + "\t" + homolog_augmentation_type + "\t" + augmentation_type + "\t" + model_type + \
        "\t" + str(replicate) + "\t" + str(sample_fraction) + \
        "\t" + str(homolog_rate) + "\t"

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
        header_line = "name\thomolog_aug_type\taug_type\tmodel\treplicate\tfraction\thomolog_rate\t"
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
