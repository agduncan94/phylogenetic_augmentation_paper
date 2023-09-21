# ####################################################################################################################
# models.py
#
# Class to train Keras models on the Basset dataset
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
from keras.models import model_from_json
from keras import backend as K
import math
import pickle
import os
import h5py
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats
import os.path
import gc
import utils
from ml_models import *

# ====================================================================================================================
# Global settings and parameters
# ====================================================================================================================
tf.debugging.set_log_device_placement(False)
ALPHABET = "ACGT"
SEQUENCE_LENGTH = 600
BATCH_SIZE = 256
TASKS = ["8988T", "AoSMC", "Chorion", "CLL", "Fibrobl", "FibroP", "Gliobla", "GM12891", "GM12892", "GM18507", "GM19238", "GM19239", "GM19240", "H9ES",
         "HeLa-S3_IFNa4h", "Hepatocytes", "HPDE6-E6E7", "HSMM_emb", "HTR8svn", "Huh-7.5", "Huh-7", "iPS", "Ishikawa_Estradiol", "Ishikawa_4OHTAM",
         "LNCaP_androgen", "MCF-7_Hypoxia", "Medullo", "Melano", "Myometr", "Osteobl", "PanIsletD", "PanIslets", "pHTE", "ProgFib", "RWPE1", "Stellate",
         "T-47D", "CD4_Th0", "Urothelia", "Urothelia_UT189", "AG04449", "AG04450", "AG09309", "AG09319", "AG10803", "AoAF", "BE2_C", "BJ", "Caco-2", "CD20+",
         "CD34+", "CMK", "GM06990", "GM12864", "GM12865", "H7-hESC", "HAc", "HAEpiC", "HA-h", "HA-sp", "HBMEC", "HCF", "HCFaa", "HCM", "HConF", "HCPEpiC", "HCT-116",
         "HEEpiC", "HFF", "HFF-Myc", "HGF", "HIPEpiC", "HL-60", "HMF", "HMVEC-dAd", "HMVEC-dBl-Ad", "HMVEC-dBl-Neo", "HMVEC-dLy-Ad", "HMVEC-dLy-Neo", "HMVEC-dNeo",
         "HMVEC-LBl", "HMVEC-LLy", "HNPCEpiC", "HPAEC", "HPAF", "HPdLF", "HPF", "HRCEpiC", "HRE", "HRGEC", "HRPEpiC", "HVMF", "Jurkat", "Monocytes-CD14+", "NB4", "NH-A",
         "NHDF-Ad", "NHDF-neo", "NHLF", "NT2-D1", "PANC-1", "PrEC", "RPTEC", "SAEC", "SKMC", "SK-N-MC", "SK-N-SH_RA", "Th2", "WERI-Rb-1", "WI-38", "WI-38_4OHTAM", "A549",
         "GM12878", "H1-hESC", "HeLa-S3", "HepG2", "HMEC", "HSMM", "HSMMtube", "HUVEC", "K562", "LNCaP", "MCF-7", "NHEK", "Th1", "LNG.IMR90", "ESC.H9", "ESC.H1",
         "IPSC.DF.6.9", "IPSC.DF.19.11", "ESDR.H1.NEUR.PROG", "ESDR.H1.BMP4.MESO", "ESDR.H1.BMP4.TROP", "ESDR.H1.MSC", "BLD.CD3.PPC", "BLD.CD3.CPC", "BLD.CD14.PC",
         "BLD.MOB.CD34.PC.M", "BLD.MOB.CD34.PC.F", "BLD.CD19.PPC", "BLD.CD56.PC", "SKIN.PEN.FRSK.FIB.01", "SKIN.PEN.FRSK.FIB.02", "SKIN.PEN.FRSK.MEL.01",
         "SKIN.PEN.FRSK.KER.02", "BRST.HMEC.35", "THYM.FET", "BRN.FET.F", "BRN.FET.M", "MUS.PSOAS", "MUS.TRNK.FET", "MUS.LEG.FET", "HRT.FET", "GI.STMC.FET",
         "GI.S.INT.FET", "GI.L.INT.FET", "GI.S.INT", "GI.STMC.GAST", "KID.FET", "LNG.FET", "OVRY", "ADRL.GLND.FET", "PLCNT.FET", "PANC"]

TRAINING = "training"
TESTING = "testing"
VALIDATION = "validation"

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# ====================================================================================================================
# Generator code for loading data from hdf5 file
# ====================================================================================================================


def get_batch(split_type, hdf5_file, seq_ids, measurements, indices, use_homologs):
    """
    Creates a batch of the input and one-hot encodes the sequences
    """

    # Retrieve sequences at the given indices
    X_batch_seqs = [seq_ids[i] for i in indices]

    # One-hot encode sequences
    X_batch = utils.one_hot_encode_batch_hdf5(
        split_type, hdf5_file, X_batch_seqs, SEQUENCE_LENGTH, use_homologs)

    # Retrieve batch of measurements
    Y_batch = measurements.iloc[indices]

    return X_batch, Y_batch


def data_gen(split_type, hdf5_file, y_file, num_samples, shuffle_epoch_end=True, use_homologs=False, order=False, filtered_indices=None):
    """
    Generator function for loading input data in batches
    """

    # Get keys from HDF5 file
    seq_ids = []
    with h5py.File(hdf5_file, "r") as f:
        for seq_id in f[split_type + '/sequences'].keys():
            seq_ids.append(seq_id)

    # Read measurement file (TODO: Use HDF5 file)
    measurements = pd.read_table(y_file)
    measurements.drop(labels=['Seq', 'Name'], axis=1, inplace=True)

    # Sample the seq ids and measurements
    if (filtered_indices is not None):
        seq_ids_filtered = [seq_ids[i] for i in filtered_indices]
        measurements = measurements.iloc[filtered_indices]
        measurements.reset_index(drop=True, inplace=True)
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
        yield get_batch(split_type, hdf5_file, seq_ids_filtered, measurements, indices[ii:ii + BATCH_SIZE], use_homologs)
        ii += BATCH_SIZE
        if ii >= num_samples:
            ii = 0
            if shuffle_epoch_end:
                if not order:
                    indices = np.random.choice(
                        list(range(num_samples)), num_samples, replace=False)
                else:
                    indices = list(range(num_samples))
        gc.collect()


# ====================================================================================================================
# Train models
# ====================================================================================================================


def train(model, model_type, use_homologs, sample_fraction, replicate, file_folder, output_folder, filtered_indices):
    """
    Train a model
    """

    # Parameters for model training
    epochs = 20
    early_stop = 10

    # Create a unique identifier for the model
    model_id = model_type + "_rep" + \
        str(replicate) + "_frac" + str(sample_fraction)

    # Create the output folder
    model_output_folder = output_folder + model_id + "/"
    os.makedirs(model_output_folder, exist_ok=True)

    # Determine the number of sequences in the train/val/test sets
    num_samples_train = utils.count_lines_in_file(
        file_folder + "Basset_Sequences_Train.txt") - 1
    num_samples_val = utils.count_lines_in_file(
        file_folder + "Basset_Sequences_Val.txt") - 1
    num_samples_test = utils.count_lines_in_file(
        file_folder + "Basset_Sequences_Test.txt") - 1

    # Print summary information about the model
    print('\n')
    print('Training model ' + model_type)
    print('Model ID: ' + model_id)
    print('Replicate: ' + str(replicate))
    print('Fraction of training data: ' + str(sample_fraction) +
          " (" + str(int(sample_fraction * num_samples_train)) + ")")
    if use_homologs:
        print('Use phylogenetic augmentations: True')
    else:
        print('Use phylogenetic augmentations: False')
    print('\n')

    # Sample a fraction of the original training data
    if int(sample_fraction) < 1:
        reduced_num_samples_train = int(num_samples_train * sample_fraction)
    else:
        reduced_num_samples_train = num_samples_train
        filtered_indices = None

    # Data generators for train and val sets used during initial training
    datagen_train = data_gen(TRAINING, file_folder + "augmentation_data_homologs.hdf5", file_folder + "Basset_Sequences_Train.txt",
                             reduced_num_samples_train, use_homologs=use_homologs, filtered_indices=filtered_indices)

    datagen_val = data_gen(VALIDATION, file_folder + "augmentation_data_homologs.hdf5", file_folder + "Basset_Sequences_Val.txt",
                           num_samples_val)

    # Fit model using the data generators
    history = model.fit(datagen_train,
                        validation_data=datagen_val,
                        epochs=epochs,
                        steps_per_epoch=math.ceil(
                            reduced_num_samples_train / BATCH_SIZE),
                        validation_steps=math.ceil(
                            num_samples_val / BATCH_SIZE),
                        callbacks=[EarlyStopping(patience=early_stop, monitor="val_loss", restore_best_weights=True),
                                   History()])

    # Save model
    if use_homologs:
        augmentation_type = 'homologs'
    else:
        augmentation_type = 'none'

    save_model(model_id + "_" + augmentation_type,
               model, history, model_output_folder)

    # Write performance metrics to file
    epochs = len(history.history['loss'])
    auc_pr = history.history['auc_pr'][epochs-1]
    validation_auc_pr = history.history['val_auc_pr'][epochs-1]
    auc_roc = history.history['auc_roc'][epochs-1]
    validation_auc_roc = history.history['val_auc_roc'][epochs-1]

    avg_auc, aucs, avg_precision, precisions, tf_aucroc, tf_auprc = plot_prediction_vs_actual(
        model, file_folder + "augmentation_data_homologs.hdf5", file_folder + "Basset_Sequences_Test.txt", model_output_folder + 'Model_' + model_id + "_" + augmentation_type + "_Test", num_samples_test, False)

    write_to_file(model_id, augmentation_type, model_type, replicate,
                  sample_fraction, history, auc_pr, validation_auc_pr, auc_roc, validation_auc_roc, aucs, avg_auc, precisions, avg_precision, tf_aucroc, tf_auprc, output_folder)

    # Save plots for performance and loss
    plot_scatterplots(history, model_output_folder,
                      model_id, augmentation_type)

    # Clear the model from memory
    del datagen_train
    del datagen_val
    reset_keras(model)
    gc.collect()


def train_basset(use_homologs, sample_fraction, replicate, file_folder, output_folder, filtered_indices=None):
    """
    Trains a Basset model on the Basset dataset
    """
    model_type = "Basset"
    input_shape, encoder = BassetEncoder(SEQUENCE_LENGTH)
    model = basset_head(input_shape, encoder, TASKS)
    train(model, model_type, use_homologs, sample_fraction, replicate,
          file_folder, output_folder, filtered_indices)


def fine_tune_basset(use_homologs, sample_fraction, replicate, file_folder, output_folder, filtered_indices=None):
    """
    Fine-tune a model
    """

    # Parameters for model fine tuning
    fine_tune_epochs = 5
    model_type = "Basset"

    # Create a unique identifier for the model
    model_id = model_type + "_rep" + \
        str(replicate) + "_frac" + str(sample_fraction)

    # Create the output folder
    model_output_folder = output_folder + model_id + "/"
    os.makedirs(model_output_folder, exist_ok=True)

    # Determine the number of sequences in the train/val/test sets
    num_samples_train = utils.count_lines_in_file(
        file_folder + "Basset_Sequences_Train.txt") - 1
    num_samples_val = utils.count_lines_in_file(
        file_folder + "Basset_Sequences_Val.txt") - 1
    num_samples_test = utils.count_lines_in_file(
        file_folder + "Basset_Sequences_Test.txt") - 1

    # Sample a reduced set of sequences for training
    if int(sample_fraction) < 1:
        reduced_num_samples_train = int(num_samples_train * sample_fraction)
    else:
        reduced_num_samples_train = num_samples_train

    # Load saved model
    if use_homologs:
        augmentation_type = 'homologs'
        augmentation_ft_type = 'homologs_finetune'
    else:
        augmentation_type = 'none'
        augmentation_ft_type = 'finetune'

    model_path_prefix = model_output_folder + \
        'Model_' + model_id + "_" + augmentation_type
    model = model_from_json(open(model_path_prefix + '.json').read())
    model.load_weights(model_path_prefix + '.h5')

    # Perform finetuning on the original training only
    model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
                  loss=['binary_crossentropy'],
                  metrics=[tf.keras.metrics.AUC(curve='PR', name="auc_pr_ft"), tf.keras.metrics.AUC(name="auc_roc_ft")])

    # Update data generator to not use homologs (not needed for fine-tuning)
    datagen_train = data_gen(TRAINING, file_folder + "augmentation_data_homologs.hdf5", file_folder + "Basset_Sequences_Train.txt",
                             reduced_num_samples_train, use_homologs=False, filtered_indices=filtered_indices)

    datagen_val = data_gen(VALIDATION, file_folder + "augmentation_data_homologs.hdf5", file_folder + "Basset_Sequences_Val.txt",
                           num_samples_val)

    # Fit the model using new generator
    fine_tune_history = model.fit(datagen_train,
                                  validation_data=datagen_val,
                                  steps_per_epoch=math.ceil(
                                      reduced_num_samples_train / BATCH_SIZE),
                                  validation_steps=math.ceil(
                                      num_samples_val / BATCH_SIZE),
                                  epochs=fine_tune_epochs)

    # Save the model
    save_model(model_id + "_" + augmentation_ft_type, model,
               fine_tune_history, model_output_folder)

    # Write performance metrics to file
    epochs = len(fine_tune_history.history['loss'])
    auc_pr = fine_tune_history.history['auc_pr_ft'][epochs-1]
    validation_auc_pr = fine_tune_history.history['val_auc_pr_ft'][epochs-1]
    auc_roc = fine_tune_history.history['auc_roc_ft'][epochs-1]
    validation_auc_roc = fine_tune_history.history['val_auc_roc_ft'][epochs-1]

    avg_auc, aucs, avg_precision, precisions, tf_aucroc, tf_auprc = plot_prediction_vs_actual(
        model, file_folder + "augmentation_data_homologs.hdf5", file_folder + "Basset_Sequences_Test.txt", model_output_folder + 'Model_' + model_id + "_" + augmentation_ft_type + "_Test", num_samples_test, False)

    write_to_file(model_id, augmentation_ft_type, model_type, replicate,
                  sample_fraction, fine_tune_history, auc_pr, validation_auc_pr, auc_roc, validation_auc_roc, aucs, avg_auc, precisions, avg_precision, tf_aucroc, tf_auprc, output_folder)

    # Save plots for performance and loss
    plot_scatterplots(fine_tune_history, model_output_folder,
                      model_id, augmentation_ft_type)

    # Clear the model from memory
    del datagen_train
    del datagen_val
    reset_keras(model)
    gc.collect()


# ====================================================================================================================
# Plot model performance for basset
# ====================================================================================================================


def plot_prediction_vs_actual(model, aug_file, activity_file, output_file_prefix, num_samples, homolog_dir, use_homologs=False):
    # Load the activity data
    Y = pd.DataFrame()

    count = 0
    for x, y in data_gen(TESTING, aug_file, activity_file, num_samples, use_homologs=use_homologs, order=True):
        Y = pd.concat((Y, y))
        count += 1
        if count >= math.ceil(num_samples / BATCH_SIZE):
            break

    # Get model predictions
    data_generator = data_gen(TESTING, aug_file, activity_file,
                              num_samples, use_homologs=use_homologs, order=True)
    Y_pred = model.predict(
        data_generator, steps=math.ceil(num_samples / BATCH_SIZE))

    # AUC ROC curve using SciKit Learn
    fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))
    aucs = []
    for i, task in enumerate(TASKS):
        fpr, tpr, thresholds = metrics.roc_curve(
            Y.iloc[:, i], Y_pred[:, i])
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' %
                  (task, auc))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')

    #print(metrics.roc_auc_score(Y.to_numpy(), Y_pred, average=None))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.savefig(output_file_prefix + "_roc.png")

    avg_auc = metrics.roc_auc_score(Y.to_numpy(), Y_pred)
    print("SKLEARN AUCROC: " + str(avg_auc))

    # Precision recall using SciKit Learn
    plt.cla()
    precision = dict()
    recall = dict()

    precisions = []

    fig, ax = plt.subplots()
    for i, task in enumerate(TASKS):
        precision[task], recall[task], _ = metrics.precision_recall_curve(Y.to_numpy()[:, i],
                                                                          Y_pred[:, i])

        avg_precision_task = metrics.average_precision_score(
            Y.to_numpy()[:, i], Y_pred[:, i])
        precisions.append(avg_precision_task)

        plt.plot(recall[task], precision[task],
                 lw=2,
                 label='%s (PR:%0.2f)' %
                 (task, avg_precision_task))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig(output_file_prefix + "_pr.png")
    plt.cla()

    avg_precision = metrics.average_precision_score(Y.to_numpy(), Y_pred)
    print("SKLEARN AUPRC: " + str(avg_precision))

    # Metrics using Tensorflow
    model_metrics = model.evaluate(data_gen(TESTING, aug_file, activity_file,
                                   num_samples), steps=math.ceil(num_samples / BATCH_SIZE))
    tf_auprc = model_metrics[1]
    tf_aucroc = model_metrics[2]

    print("TF AUC ROC: " + str(tf_aucroc))
    print("TF AUPRC: " + str(tf_auprc))

    return avg_auc, aucs, avg_precision, precisions, tf_aucroc, tf_auprc


def plot_scatterplot(history, a, b, x, y, title, filename):
    """Plots a scatterplot and saves to file"""
    fig, ax = plt.subplots()
    plt.plot(history.history[a])
    plt.plot(history.history[b])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(filename)
    plt.cla()


def plot_scatterplots(history, model_output_folder, model_id, name):
    """Plots model performance and loss for each task of a given model"""
    plot_scatterplot(history, 'loss', 'val_loss', 'epoch', 'loss',
                     'Model loss', model_output_folder + 'Model_' + model_id + '_' + name + '_loss.png')

# ====================================================================================================================
# Helpers
# ====================================================================================================================


def write_to_file(model_id, augmentation_type, model_type, replicate, sample_fraction, history, auc_pr, validation_auc_pr, auc_roc, validation_auc_roc, test_auc, mean_test_auc, test_pr, mean_test_pr, tf_aucroc, tf_auprc, output_folder):
    """Writes model performance to a file"""

    correlation_file_path = output_folder + 'model_metrics.tsv'

    # Generate line to write to file
    line = model_id + "\t" + augmentation_type + "\t" + model_type + \
        "\t" + str(replicate) + "\t" + str(sample_fraction) + \
        "\t" + str(auc_pr) + "\t" + str(validation_auc_pr) + \
        "\t" + str(auc_roc) + "\t" + str(validation_auc_roc) + \
        "\t" + str(mean_test_auc) + "\t" + str(mean_test_pr) + \
        "\t" + str(tf_aucroc) + "\t" + str(tf_auprc) + "\t"

    for i, task in enumerate(TASKS):
        line += str(test_auc[i]) + "\t"

    for i, task in enumerate(TASKS):
        if i == len(TASKS) - 1:
            line += str(test_pr[i]) + "\n"
        else:
            line += str(test_pr[i]) + "\t"

    # Write line to file (and also header if necessary)
    if os.path.isfile(correlation_file_path):
        f = open(correlation_file_path, "a")
        f.write(line)
        f.close()
    else:
        f = open(correlation_file_path, "w")
        header_line = "name\ttype\tmodel\treplicate\tfraction\ttrain_auc_pr\tval_auc_pr\ttrain_auc_roc\tval_auc_roc\tmean_test_auc\tmean_test_pr\ttf_aucroc\ttf_auprc\t"

        for i, task in enumerate(TASKS):
            header_line += "test_auc_" + task + "\t"

        for i, task in enumerate(TASKS):
            if i == len(TASKS) - 1:
                header_line += "test_pr_" + task + "\n"
            else:
                header_line += "test_pr_" + task + "\t"

        f.write(header_line)
        f.write(line)
        f.close()
