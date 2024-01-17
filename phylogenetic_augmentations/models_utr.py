# ####################################################################################################################
# models_utr.py
#
# Class to train Keras models on yeast 3' UTRs
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
from keras.callbacks import EarlyStopping, History
from keras import backend as K
import math
import os
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats
import os.path
import utils
from ml_models import *
from sklearn import metrics

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
tf.debugging.set_log_device_placement(False)
ALPHABET = "ACGT"
SEQUENCE_LENGTH = 200
BATCH_SIZE = 256
TASKS = ['PUF3']

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# ====================================================================================================================
# Generator code for loading data
# ====================================================================================================================

def get_batch(homolog_obj, data, indices, use_homologs=False, phylo_aug_rate=1.0):
    """
    Creates a batch of the input and one-hot encodes the sequences
    """

    # One-hot encode a batch of sequences
    X_batch = homolog_obj.one_hot_encode_batch(
        indices, SEQUENCE_LENGTH, use_homologs, phylo_aug_rate, False)

    # Retrieve batch of measurements
    Y = []
    for i, task in enumerate(TASKS):
        Y.append((data[data.columns[i]][indices]).values)

    # Shuffle
    Y_batch = []
    for i, task in enumerate(TASKS):
        col_val = Y[i]
        Y_batch.append(col_val)

    # Create final output
    return X_batch, (np.array(Y_batch)).transpose()


def data_gen(input_file, homolog_folder, num_samples, shuffle_epoch_end=True, use_homologs=False, species=None, order=False, filtered_indices=None, phylo_aug_rate=1.0):
    """
    Generator function for loading input data in batches
    """

    # Read input file into memory
    data = pd.read_table(input_file)

    # Optionally sample a fraction of the input data
    if filtered_indices is not None:
        data = data.iloc[filtered_indices]
        data.reset_index(drop=True, inplace=True)

    # Create in memory object with homolog sequences
    homolog_obj = utils.homolog_fastas(data)

    # Add homologs to FASTA data structure
    if use_homologs:
        directory = os.fsencode(homolog_folder)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".fa") and (species is None or [ele for ele in species if(ele in filename)]):
                #print(filename)
                homolog_obj.add_homolog_sequences(
                    os.path.join(homolog_folder, filename))

    # Create the batch indices
    if not order:
        indices = np.random.choice(
            list(range(num_samples)), num_samples, replace=False)
    else:
        indices = list(range(num_samples))

    ii = 0
    while True:
        yield get_batch(homolog_obj, data, indices[ii:ii + BATCH_SIZE], use_homologs, phylo_aug_rate)
        ii += BATCH_SIZE
        if ii >= num_samples:
            ii = 0
            if shuffle_epoch_end:
                if not order:
                    indices = np.random.choice(
                        list(range(num_samples)), num_samples, replace=False)
                else:
                    indices = list(range(num_samples))


# ====================================================================================================================
# Train models
# ====================================================================================================================


def train(model, model_type, use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, phylo_aug_rate=1.0, species=None):
    """
    Train a model
    """

    # Parameters for model training
    epochs = 50
    fine_tune_epochs = 5

    # Create a unique identifier for the model
    model_id = model_type + "_rep" + \
        str(replicate) + "_frac" + str(sample_fraction)

    # Create the output folder
    model_output_folder = output_folder + model_id + "/"
    os.makedirs(model_output_folder, exist_ok=True)

    # Determine the number of sequences in the train/test sets (subtract 1 for header row)
    train_file = file_folder + "Yeast_Sequences_Train.txt"
    test_file = file_folder + "Yeast_Sequences_Test.txt"

    num_samples_train = utils.count_lines_in_file(
        train_file) - 1
    num_samples_test = utils.count_lines_in_file(
        test_file) - 1

    # Print summary information about the model
    print('\n')
    print('Training model ' + model_type)
    print('============================================================')
    print('Model ID: ' + model_id)
    print('Replicate: ' + str(replicate))
    print('Fraction of training data: ' + str(sample_fraction) +
          " (" + str(num_samples_train) + ")")
    if use_homologs:
        print('Use phylogenetic augmentations: True')
        print('Phylogenetic augmentation rate: ' + str(phylo_aug_rate))
    else:
        print('Use phylogenetic augmentations: False')

    if species is not None:
        print('Species: ' + str(len(species)))
    print('\n')

    # Sample a fraction of the original training data (if specified)
    reduced_num_samples_train = int(num_samples_train * sample_fraction)
    filtered_indices = np.random.choice(
        list(range(num_samples_train)), reduced_num_samples_train, replace=False)

    # Data generators for training set used during model training
    datagen_train = data_gen(train_file,
                             homolog_folder, reduced_num_samples_train, use_homologs=use_homologs, species=species, filtered_indices=filtered_indices, phylo_aug_rate=phylo_aug_rate)

    # Fit model using the data generators
    history = model.fit(datagen_train,
                        epochs=epochs,
                        steps_per_epoch=math.ceil(
                            reduced_num_samples_train / BATCH_SIZE),
                        callbacks=[History()])

    # Define augmentation type
    if use_homologs:
        augmentation_type = 'homologs'
    else:
        augmentation_type = 'none'

    # Save model
    save_model(model_id + "_" + augmentation_type,
               model, history, model_output_folder)

    # Plot test performance on a scatterplot
    test_metrics = plot_prediction_vs_actual(model, test_file,
                                                  model_output_folder + 'Model_' + model_id + "_" +
                                                  augmentation_type + "_Test",
                                                  num_samples_test,
                                                  homolog_folder,
                                                  False)
    training_metrics = get_performance_metrics(model, train_file, num_samples_train, homolog_folder, False) 

    # Write performance metrics to file
    write_to_file(model_id, augmentation_type, model_type, replicate,
                  sample_fraction, history, training_metrics, test_metrics, phylo_aug_rate, species, output_folder)

    # Save plots for performance and loss
    plot_scatterplots(history, model_output_folder,
                      model_id, augmentation_type)

    # Perform fine-tuning on the original training only
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
                  loss=['binary_crossentropy'],
                  metrics=['Accuracy', tf.keras.metrics.AUC(multi_label=True, curve="PR", num_labels=len(TASKS))])

    # Update data generator to not use homologs (not needed for fine-tuning)
    if use_homologs:
        datagen_train = data_gen(train_file,
                                 homolog_folder, reduced_num_samples_train, filtered_indices=filtered_indices)

    # Fit the model using new generator
    fine_tune_history = model.fit(datagen_train,
                                  steps_per_epoch=math.ceil(
                                      reduced_num_samples_train / BATCH_SIZE),
                                  epochs=fine_tune_epochs)
    print(fine_tune_history.history)
    # Save model
    if use_homologs:
        augmentation_ft_type = 'homologs_finetune'
    else:
        augmentation_ft_type = 'finetune'

    save_model(model_id + "_" + augmentation_ft_type, model,
               fine_tune_history, model_output_folder)

    # Plot test performance on a scatterplot
    test_metrics = plot_prediction_vs_actual(model, test_file,
                                                  model_output_folder + 'Model_' + model_id +
                                                  "_" + augmentation_ft_type + "_Test",
                                                  num_samples_test,
                                                  homolog_folder,
                                                  False)
    training_metrics = get_performance_metrics(model, train_file, num_samples_train, homolog_folder, False) 

    # Write performance metrics to file
    write_to_file(model_id, augmentation_ft_type, model_type, replicate,
                  sample_fraction, fine_tune_history, training_metrics, test_metrics, phylo_aug_rate, species, output_folder)

    # Save plots for performance and loss
    plot_scatterplots(fine_tune_history, model_output_folder,
                      model_id, augmentation_ft_type)

    # Clean up
    clear_keras(model)


def train_motif_deepstarr(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, phylo_aug_rate=1.0, species=None):
    """
    Trains a simplified Motif DeepSTARR model on the UTR binding sites
    """
    model_type = "SimpleMotifDeepSTARR"
    input_shape, encoder = SimplifiedMotifDeepSTARREncoder(SEQUENCE_LENGTH)
    model = n_classification_head(input_shape, encoder, TASKS)
    #model.summary()
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, phylo_aug_rate, species)

def train_deepstarr(use_homologs, sample_fraction, replicate, file_folder, homolog_folder, output_folder, phylo_aug_rate=1.0, species=None):
    """
    Trains a DeepSTARR model on the UTR binding sites
    """
    model_type = "Scrambled Control"
    input_shape, encoder = DeepSTARREncoder(SEQUENCE_LENGTH)
    model = n_classification_head(input_shape, encoder, TASKS)
    #model.summary()
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, homolog_folder, output_folder, phylo_aug_rate, species)
    
# ====================================================================================================================
# Plot model performance for dual regression
# ====================================================================================================================


def plot_prediction_vs_actual(model, input_file, output_file_prefix, num_samples, homolog_folder, use_homologs=False):
    """
    Plots the predicted vs actual activity for each task on given input set
    """

    # Load the actual labels
    Y = []

    count = 0
    for x, y in data_gen(input_file, homolog_folder, num_samples, use_homologs=use_homologs, order=True):
        if count == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))
        count += 1
        if count > (num_samples / BATCH_SIZE):
            break

    # Retrieve model predictions
    data_generator = data_gen(input_file, homolog_folder,
                              num_samples, use_homologs=use_homologs, order=True)
    
    model_metrics = model.evaluate(
        data_generator, steps=math.ceil(num_samples / BATCH_SIZE))
    
    # Plot confusion matrix
    Y_pred = model.predict(
        data_generator, steps=math.ceil(num_samples / BATCH_SIZE))

    Y_transposed = (np.array(Y)).transpose()
    Y_pred_transposed = (np.array(Y_pred)).transpose()

    for i, task in enumerate(TASKS):
        Y_pred_task = Y_pred_transposed[i]
        Y_task = Y_transposed[i]
        Y_pred_task = np.rint(Y_pred_task)

    return model_metrics


def get_performance_metrics(model, input_file, num_samples, homolog_folder, use_homologs=False):
    data_generator = data_gen(input_file, homolog_folder,
                              num_samples, use_homologs=use_homologs, order=True)
    
    model_metrics = model.evaluate(
        data_generator, steps=math.ceil(num_samples / BATCH_SIZE))

    return model_metrics



def plot_scatterplot(history, a, x, y, title, filename):
    """
    Plots a scatterplot and saves to file
    """
    fig, ax = plt.subplots()
    plt.plot(history.history[a])
    plt.title(title)
    plt.ylabel(x)
    plt.xlabel(y)
    plt.legend(['train'], loc='upper left')
    plt.savefig(filename)
    plt.cla()
    plt.close()


def plot_scatterplots(history, model_output_folder, model_id, name):
    """
    Plots model performance and loss for each task of a given model
    """
    plot_scatterplot(history, 'loss', 'loss', 'epoch',
                        'Model loss', model_output_folder + 'Model_' + model_id + '_' + name + '_loss.png')

# ====================================================================================================================
# Helpers
# ====================================================================================================================


def write_to_file(model_id, augmentation_type, model_type, replicate, sample_fraction, history, training_metrics, test_metrics, phylo_aug_rate, species, output_folder):
    """
    Writes model performance to a file
    """

    correlation_file_path = output_folder + 'model_metrics.tsv'

    if species is None:
        species = 0
    else:
        species = len(species)

    # Generate line to write to file
    line = model_id + "\t" + str(augmentation_type) + "\t" + str(model_type) + \
        "\t" + str(replicate) + "\t" + str(sample_fraction) + \
        "\t" + str(phylo_aug_rate) + "\t" + str(species) + \
        '\t' + str(training_metrics[1]) + \
        '\t' + str(test_metrics[1]) + \
        '\t' + str(test_metrics[2]) + '\n'

    # Write line to file (and also header if necessary)
    if os.path.isfile(correlation_file_path):
        f = open(correlation_file_path, "a")
        f.write(line)
        f.close()
    else:
        f = open(correlation_file_path, "w")
        header_line = "name\ttype\tmodel\treplicate\tfraction\tphylo_aug_rate\tspecies\taccuracy_train\taccuracy_test\tpr_test\n"
        f.write(header_line)
        f.write(line)
        f.close()
