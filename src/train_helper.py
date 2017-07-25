# -*- coding: utf-8 -*-
"""Training script for planet amazon deforestation kaggle."""
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.append('../src')
sys.path.append('../tests')

# <markdowncell>

# ## Import required modules

# <codecell>

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


import data_helper
import planet_kaggle_helper

from keras_transfer_learning import TransferModel
from keras_helper import AmazonKerasClassifier as CNNModel

from kaggle_data.downloader import KaggleDataDownloader
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import fbeta_score

import logging
import time

from itertools import chain
from sklearn.model_selection import train_test_split

# Setup logging to std out
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug("TensorFlow version: " + tf.__version__)

DATA_FORMAT = planet_kaggle_helper.DATA_FORMAT


def run(annealing, architecture, batch_size, best_cnn_weights_path, best_full_weights_path, best_retrain_weights_path,
        best_top_weights_path, cnn_epochs_arr, cnn_learn_rates, eval, fine_epochs_arr, fine_learn_rates,
        fine_momentum_arr, generate_test, generate_test_ensemble, img_size, load, load_cnn_model, load_cnn_weights_path,
        load_full_weights_path, load_top_weights_path, max_train_time_hrs, n_classes, n_untrained_layers,
        top_epochs_arr, top_learn_rates, train, train_cnn, train_top, validation_split_size):
    logger.info("img_size: " + str(img_size))
    logger.info("top_epochs_arr: " + str(top_epochs_arr))
    logger.info("top_learn_rates: " + str(top_learn_rates))
    logger.info("cnn_epochs_arr: " + str(cnn_epochs_arr))
    logger.info("cnn_learn_rates: " + str(cnn_learn_rates))
    logger.info("n_untrained_layers: " + str(n_untrained_layers))
    logger.info("fine_epochs_arr: " + str(fine_epochs_arr))
    logger.info("fine_learn_rates: " + str(fine_learn_rates))
    logger.info("fine_momentum_arr: " + str(fine_momentum_arr))

    classifiers = []

    # get data paths from competition helper
    train_processed_dir, test_processed_dir = planet_kaggle_helper.get_proccessed_data_paths()
    train_dir, test_dir, test_additional, train_csv_file = planet_kaggle_helper.get_data_files_paths()

    if train or train_top or train_cnn or eval or generate_test or generate_test_ensemble:
        x_input, y_true, y_map = data_helper.load_train_input(img_size, train_dir, train_csv_file, train_processed_dir)
        logger.debug("x_input shape: {}".format(x_input.shape))
        logger.debug("y_true shape: {}".format(y_true.shape))
        logger.debug("Label mapping: " + str(y_map))

    if train_top:
        X_train, X_valid, y_train, y_valid = train_test_split(x_input, y_true,
                                                              test_size=validation_split_size)

        classifier = train_top_model(img_size, batch_size, best_top_weights_path, top_epochs_arr,
                                     top_learn_rates, validation_split_size, X_train, X_valid, y_train, y_valid)
    if train:
        if n_untrained_layers == 0:
            classifier = retrain_model(architecture, x_input, y_true, img_size, annealing, batch_size,
                                       best_retrain_weights_path,
                                       best_top_weights_path, fine_epochs_arr, fine_learn_rates, fine_momentum_arr,
                                       max_train_time_hrs, top_epochs_arr, top_learn_rates,
                                       validation_split_size)
        else:
            classifier = fine_tune_model(architecture, x_input, y_true, img_size, annealing, batch_size,
                                         best_full_weights_path,
                                         best_top_weights_path, fine_epochs_arr, fine_learn_rates, fine_momentum_arr,
                                         max_train_time_hrs, n_untrained_layers, top_epochs_arr, top_learn_rates,
                                         validation_split_size)
    if train_cnn:
        X_train, X_valid, y_train, y_valid = train_test_split(x_input, y_true,
                                                              test_size=validation_split_size)

        classifier = train_cnn_model(img_size, batch_size, best_cnn_weights_path, cnn_epochs_arr, cnn_learn_rates,
                                     validation_split_size, X_train, X_valid, y_train, y_valid)
    if load:
        for architecture, n_untrained_layers, load_top_weights_path, load_full_weights_path in zip(architecture,
                                                                                                   n_untrained_layers,
                                                                                                   load_top_weights_path,
                                                                                                   load_full_weights_path):
            if n_untrained_layers == 0:
                classifier = load_retrained_model(architecture, img_size, n_classes, load_top_weights_path,
                                                  load_full_weights_path)
            else:
                classifier = load_fine_tuned_model(architecture, img_size, n_classes, n_untrained_layers,
                                                   load_top_weights_path, load_full_weights_path)
            classifiers.append(classifier)

    if load_cnn_model:
        classifier = load_cnn(img_size, n_classes, load_cnn_weights_path)
        classifiers.append(classifier)

    if eval:
        f2, threshold = evaluate(classifier, x_input, y_true)
        logger.info("WARNING: This eval is a rough sanity check, it will include training data.")
        logger.info("F2(c_thresh=" + str(threshold) + "): " + str(f2))

    if generate_test:
        del x_input
        del y_true
        gc.collect()
        x_test, x_test_filename = data_helper.load_test_input(img_size, test_dir, test_additional, test_processed_dir)
        logger.debug("x_test shape: {}".format(x_test.shape))
        logger.debug("x_test_filename length: {}".format(len(x_test_filename)))
        create_submission_file(classifier, x_test, x_test_filename, y_map)

    if generate_test_ensemble:
        del x_input
        del y_true
        gc.collect()
        x_test, x_test_filename = data_helper.load_test_input(img_size, test_dir, test_additional, test_processed_dir)
        logger.debug("x_test shape: {}".format(x_test.shape))
        logger.debug("x_test_filename length: {}".format(len(x_test_filename)))
        create_submission_file_ensemble(classifiers, x_test, x_test_filename, y_map)

    return None


def fine_tune_model(architecture, x_input, y_true, img_size, annealing, batch_size, best_full_weights_path, best_top_weights_path,
                    fine_epochs_arr, fine_learn_rates, fine_momentum_arr, max_train_time_hrs, n_untrained_layers,
                    top_epochs_arr, top_learn_rates, validation_split_size):
    X_train, X_valid, y_train, y_valid = train_test_split(x_input, y_true,
                                                          test_size=validation_split_size)

    classifier = train_top_model(architecture, img_size, batch_size, best_top_weights_path, top_epochs_arr,
                                 top_learn_rates, validation_split_size, X_train, X_valid, y_train, y_valid)

    # ## Fine-tune full model
    #
    # Now we retrain the top model together with the base model
    classifier = train_partial_model(architecture, batch_size, best_full_weights_path, classifier, fine_epochs_arr,
                                     fine_learn_rates, fine_momentum_arr, max_train_time_hrs, n_untrained_layers,
                                     validation_split_size, X_train, X_valid, y_train, y_valid, annealing)
    del X_train
    del X_valid
    del y_train
    del y_valid
    gc.collect()
    return classifier


def retrain_model(architecture, x_input, y_true, img_size, annealing, batch_size, best_retrain_weights_path, best_top_weights_path,
                    fine_epochs_arr, fine_learn_rates, fine_momentum_arr, max_train_time_hrs,
                    top_epochs_arr, top_learn_rates, validation_split_size):
    X_train, X_valid, y_train, y_valid = train_test_split(x_input, y_true,
                                                          test_size=validation_split_size)

    classifier = train_top_model(architecture, img_size, batch_size, best_top_weights_path, top_epochs_arr,
                                 top_learn_rates, validation_split_size, X_train, X_valid, y_train, y_valid)

    # ## Fine-tune full model
    #
    # Now we retrain the top model together with the base model
    classifier = retrain_full_model(architecture, batch_size, best_retrain_weights_path, classifier, fine_epochs_arr,
                                     fine_learn_rates, fine_momentum_arr, max_train_time_hrs,
                                     validation_split_size, X_train, X_valid, y_train, y_valid, annealing)
    del X_train
    del X_valid
    del y_train
    del y_valid
    gc.collect()
    return classifier


def load_fine_tuned_model(architecture, img_size, n_classes, n_untrained_layers, top_weights_path, fine_weights_path):
    classifier = TransferModel()
    classifier.build_base_model(architecture, [img_size, img_size], 3)
    classifier.build_top_model(n_classes)
    classifier.load_top_weights(top_weights_path)
    classifier.split_fine_tuning_models(n_untrained_layers)
    classifier.load_top_weights(fine_weights_path)
    logger.debug("Loaded " + architecture +" model.")
    return classifier

def load_retrained_model(architecture, img_size, n_classes, top_weights_path, fine_weights_path):
    classifier = TransferModel()
    classifier.build_base_model(architecture, [img_size, img_size], 3)
    classifier.build_top_model(n_classes)
    classifier.load_top_weights(top_weights_path)
    classifier.set_full_retrain()
    classifier.load_top_weights(fine_weights_path)
    logger.debug("Loaded " + architecture +" model.")
    return classifier

def train_partial_model(architecture, batch_size, best_full_weights_path, classifier, fine_epochs_arr,
                        fine_learn_rates, fine_momentum_arr, max_train_time_hrs, n_untrained_layers,
                        validation_split_size, X_train, X_valid, y_train, y_valid, annealing):

    # Create a checkpoint, for best model weights
    checkpoint_full = ModelCheckpoint(best_full_weights_path, monitor='val_loss', verbose=1, save_best_only=True)

    max_train_time_secs = max_train_time_hrs * 60 * 60

    logger.info("Fine tuning top model and base model layers.")
    logger.info("Will train for max " + str(float(max_train_time_secs) / 60) + " min.")
    init_top_weights = classifier.split_fine_tuning_models(n_untrained_layers)
    split_layer_name = classifier.base_model.layers[n_untrained_layers].name
    logger.info("Splitting at: " + split_layer_name + "(last base model layer)")
    classifier.predict_bottleneck_features(X_train, X_valid)

    logger.info("Bottleneck features calculated.")
    train_losses_full, val_losses_full = [], []
    start = time.time()
    first_flag = True
    for learn_rate, epochs, momentum in zip(fine_learn_rates, fine_epochs_arr, fine_momentum_arr):
        if not annealing:
            #reload init weights for new experiment run
            classifier.set_top_weights(init_top_weights)
        else:
            if not first_flag:
                #reload best weights from previous run for new annealing run
                classifier.load_top_weights(best_full_weights_path)
            else:
                #first annealing run
                first_flag = False

        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.fine_tune_full_model(y_train, y_valid, learn_rate,
                                                                                        momentum, epochs,
                                                                                        batch_size,
                                                                                        train_callbacks=[
                                                                                            checkpoint_full])

        train_losses_full += tmp_train_losses
        val_losses_full += tmp_val_losses
        logger.info("learn_rate : " + str(learn_rate))
        logger.info("epochs : " + str(epochs))
        logger.info("momentum : " + str(momentum))
        logger.info("fbeta_score : " + str(fbeta_score))
        logger.info("classification_threshold : " + str(classifier.classification_threshold))

        curr_time = time.time()
        train_duration = curr_time - start
        if train_duration > max_train_time_secs:
            logger.info("Training canceled due to max train time parameter.")
            break
        else:
            logger.debug("Keep training: " + str(train_duration) + " < " + str(max_train_time_secs))
    end = time.time()
    t_epoch = float(end - start) / sum(fine_epochs_arr)
    logger.info("Training time [min]: " + str(float(end - start) / 60))
    logger.info("Training time [s/epoch]: " + str(t_epoch))
    # ## Load Best Weights
    # Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training
    classifier.load_top_weights(best_full_weights_path)
    # ## Monitor the results
    # Check that we do not overfit by plotting the losses of the train and validation sets
    # plt.plot(train_losses_full, label='Training loss')
    # plt.plot(val_losses_full, label='Validation loss')
    # plt.legend();
    # Store losses
    np.save("fine_train_losses.npy", train_losses_full)
    np.save("fine_val_losses.npy", val_losses_full)
    plt.plot(train_losses_full, label='Training loss')
    plt.plot(val_losses_full, label='Validation loss')
    plt.legend()
    plt.savefig('fine_loss.png')
    # Look at our fbeta_score
    fbeta_score = classifier.get_fbeta_score_valid(y_valid)
    logger.info("Best fine-tuning F2: " + str(fbeta_score))

    return classifier

def retrain_full_model(architecture, batch_size, best_retrain_weights_path, classifier, fine_epochs_arr,
                        fine_learn_rates, fine_momentum_arr, max_train_time_hrs,
                        validation_split_size, X_train, X_valid, y_train, y_valid, annealing):

    # Create a checkpoint, for best model weights
    checkpoint_full = ModelCheckpoint(best_retrain_weights_path, monitor='val_loss', verbose=1, save_best_only=True)

    max_train_time_secs = max_train_time_hrs * 60 * 60

    logger.info("Retraining top model and base model layers.")
    logger.info("Will train for max " + str(float(max_train_time_secs) / 60) + " min.")
    init_top_weights = classifier.set_full_retrain()

    train_losses_full, val_losses_full = [], []
    start = time.time()
    first_flag = True
    for learn_rate, epochs, momentum in zip(fine_learn_rates, fine_epochs_arr, fine_momentum_arr):
        if not annealing:
            #reload init weights for new experiment run
            classifier.set_top_weights(init_top_weights)
        else:
            if not first_flag:
                #reload best weights from previous run for new annealing run
                classifier.load_top_weights(best_retrain_weights_path)
            else:
                #first annealing run
                first_flag = False

        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.retrain_full_model(X_train, X_valid, y_train, y_valid, learn_rate,
                                                                                        momentum, epochs,
                                                                                        batch_size,
                                                                                        train_callbacks=[
                                                                                            checkpoint_full])

        train_losses_full += tmp_train_losses
        val_losses_full += tmp_val_losses
        logger.info("learn_rate : " + str(learn_rate))
        logger.info("epochs : " + str(epochs))
        logger.info("momentum : " + str(momentum))
        logger.info("fbeta_score : " + str(fbeta_score))
        logger.info("classification_threshold : " + str(classifier.classification_threshold))

        curr_time = time.time()
        train_duration = curr_time - start
        if train_duration > max_train_time_secs:
            logger.info("Training canceled due to max train time parameter.")
            break
        else:
            logger.debug("Keep training: " + str(train_duration) + " < " + str(max_train_time_secs))
    end = time.time()
    t_epoch = float(end - start) / sum(fine_epochs_arr)
    logger.info("Training time [min]: " + str(float(end - start) / 60))
    logger.info("Training time [s/epoch]: " + str(t_epoch))
    # ## Load Best Weights
    # Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training
    classifier.load_top_weights(best_retrain_weights_path)
    # ## Monitor the results
    # Check that we do not overfit by plotting the losses of the train and validation sets
    # plt.plot(train_losses_full, label='Training loss')
    # plt.plot(val_losses_full, label='Validation loss')
    # plt.legend();
    # Store losses
    np.save("retrain_train_losses.npy", train_losses_full)
    np.save("retrain_val_losses.npy", val_losses_full)
    plt.plot(train_losses_full, label='Training loss')
    plt.plot(val_losses_full, label='Validation loss')
    plt.legend()
    plt.savefig('retrain_loss.png')
    # Look at our fbeta_score
    fbeta_score = classifier._get_fbeta_score(classifier, X_valid, y_valid)
    logger.info("Best retraining F2: " + str(fbeta_score))

    return classifier


def train_top_model(architecture, img_size, batch_size, best_top_weights_path, top_epochs_arr, top_learn_rates,
                    validation_split_size, X_train, X_valid, y_train, y_valid):
    # Create a checkpoint, for best model weights
    checkpoint_top = ModelCheckpoint(best_top_weights_path, monitor='val_loss', verbose=1, save_best_only=True)

    img_resize = (img_size, img_size)

    # Define and Train model
    n_classes = y_train.shape[1]

    logger.info("Training dense top model.")
    classifier = TransferModel()
    logger.info("Classifier initialized.")
    classifier.build_base_model(architecture, img_resize, 3)
    logger.info("Base model " + architecture + " built.")
    classifier.predict_bottleneck_features(X_train, X_valid)

    logger.info("Bottleneck features calculated.")
    classifier.build_top_model(n_classes)
    logger.info("Top built, ready to train.")
    train_losses, val_losses = [], []
    start = time.time()
    for learn_rate, epochs in zip(top_learn_rates, top_epochs_arr):
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_top_model(y_train, y_valid, learn_rate, epochs,
                                                                                   batch_size,
                                                                                   train_callbacks=[checkpoint_top])
        train_losses += tmp_train_losses
        val_losses += tmp_val_losses

        logger.info("learn_rate : " + str(learn_rate))
        logger.info("epochs : " + str(epochs))
        logger.info("fbeta_score : " + str(fbeta_score))
        logger.info("classification_threshold : " + str(classifier.classification_threshold))

    end = time.time()
    t_epoch = float(end - start) / sum(top_epochs_arr)
    logger.info("Training time [s/epoch]: " + str(t_epoch))
    # ## Load Best Weights
    # Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training
    classifier.load_top_weights(best_top_weights_path)
    logger.info("Weights loaded")
    # ## Monitor the results
    # Check that we do not overfit by plotting the losses of the train and validation sets
    # plt.plot(train_losses, label='Training loss')
    # plt.plot(val_losses, label='Validation loss')
    # plt.legend();
    # Store losses
    np.save("top_train_losses.npy", train_losses)
    np.save("top_tval_losses.npy", val_losses)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.savefig('top_loss.png')
    # Look at our fbeta_score
    fbeta_score = classifier.get_fbeta_score_valid(y_valid)
    logger.info("Best top model F2: " + str(fbeta_score))
    return classifier


def load_cnn(img_size, n_classes, best_cnn_weights_path):
    img_resize = (img_size, img_size)
    classifier = CNNModel()
    classifier.add_conv_layer(img_resize)
    classifier.add_flatten_layer()
    classifier.add_ann_layer(n_classes)
    classifier.load_weights(best_cnn_weights_path)
    logger.debug("Loaded CNN model.")
    return classifier


def train_cnn_model(img_size, batch_size, best_cnn_weights_path, cnn_epochs_arr, cnn_learn_rates,
                    validation_split_size, X_train, X_valid, y_train, y_valid):
    # Create a checkpoint, for best model weights
    checkpoint_cnn = ModelCheckpoint(best_cnn_weights_path, monitor='val_loss', verbose=1, save_best_only=True)

    img_resize = (img_size, img_size)
    n_classes = y_train.shape[1]

    # Define and Train model
    classifier = CNNModel()
    classifier.add_conv_layer(img_resize)
    classifier.add_flatten_layer()
    classifier.add_ann_layer(n_classes)
    logger.info("CNN built, ready to train.")
    train_losses, val_losses = [], []
    start = time.time()
    for learn_rate, epochs in zip(cnn_learn_rates, cnn_epochs_arr):
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model_new(X_train, X_valid, y_train, y_valid, learn_rate, epochs,
                                                                                   batch_size,
                                                                                   validation_split_size=validation_split_size,
                                                                                   train_callbacks=[checkpoint_cnn])
        train_losses += tmp_train_losses
        val_losses += tmp_val_losses

        logger.info("learn_rate : " + str(learn_rate))
        logger.info("epochs : " + str(epochs))
        logger.info("fbeta_score : " + str(fbeta_score))
        logger.info("classification_threshold : " + str(classifier.classification_threshold))

    end = time.time()
    t_epoch = float(end - start) / sum(cnn_epochs_arr)
    logger.info("Training time [s/epoch]: " + str(t_epoch))
    # ## Load Best Weights
    # Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training
    classifier.load_weights(best_cnn_weights_path)
    logger.info("Weights loaded")
    # ## Monitor the results
    # Check that we do not overfit by plotting the losses of the train and validation sets
    # plt.plot(train_losses, label='Training loss')
    # plt.plot(val_losses, label='Validation loss')
    # plt.legend();
    # Store losses
    np.save("cnn_train_losses.npy", train_losses)
    np.save("cnn_val_losses.npy", val_losses)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.savefig('cnn_loss.png')
    # Look at our fbeta_score
    fbeta_score = classifier._get_fbeta_score(classifier, X_valid, y_valid)
    logger.info("Best CNN model F2: " + str(fbeta_score))
    return classifier

def create_submission_file(classifier, x_test, x_test_filename , y_map, classification_threshold=0.2):

    predictions = classifier.predict(x_test)
    logger.info("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape,
                                                                                           x_test_filename.shape,
                                                                                           predictions[0]))

    thresholds = [classification_threshold] * len(y_map)

    # TODO complete
    #tags_pred = np.array(predictions).T
    #_, axs = plt.subplots(5, 4, figsize=(15, 20))
    #axs = axs.ravel()

    #for i, tag_vals in enumerate(tags_pred):
    #    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_map[i])

    # Now lets map our predictions to their tags and use the thresholds we just retrieved
    predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)

    # Finally lets assemble and visualize our prediction for the test dataset
    tags_list = [None] * len(predicted_labels)
    for i, tags in enumerate(predicted_labels):
        tags_list[i] = ' '.join(map(str, tags))

    final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]


    final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
    #final_df.head()


    #tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
    #fig, ax = plt.subplots(figsize=(16, 8))
    #sns.barplot(x=tags_s, y=tags_s.index, orient=' h');

    # And save it to a submission file
    final_df.to_csv('../submission_file.csv', index=False)
    classifier.close()

def create_submission_file_ensemble(classifiers, x_test, x_test_filename , y_map, classification_threshold=0.2):
    predictions = None
    for classifier in classifiers:
        predictions_tmp = classifier.predict(x_test)
        if predictions is None:
            predictions = predictions_tmp
        else:
            predictions += predictions_tmp

    predictions = predictions / len(classifiers)
    logger.info("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape,
                                                                                           x_test_filename.shape,
                                                                                           predictions[0]))

    thresholds = [classification_threshold] * len(y_map)

    # TODO complete
    #tags_pred = np.array(predictions).T
    #_, axs = plt.subplots(5, 4, figsize=(15, 20))
    #axs = axs.ravel()

    #for i, tag_vals in enumerate(tags_pred):
    #    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_map[i])

    # Now lets map our predictions to their tags and use the thresholds we just retrieved
    predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)

    # Finally lets assemble and visualize our prediction for the test dataset
    tags_list = [None] * len(predicted_labels)
    for i, tags in enumerate(predicted_labels):
        tags_list[i] = ' '.join(map(str, tags))

    final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]


    final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
    #final_df.head()


    #tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
    #fig, ax = plt.subplots(figsize=(16, 8))
    #sns.barplot(x=tags_s, y=tags_s.index, orient=' h');

    # And save it to a submission file
    final_df.to_csv('../submission_file.csv', index=False)
    classifier.close()


def evaluate(classifier, x_input, y_true, threshold=None):
    """Call f2 on prediction of input and gold labels."""
    prediction = classifier.predict(x_input)
    if threshold is None:
        threshold = classifier.classification_threshold
        if threshold is None:
            threshold = 0.2
    f2 = fbeta_score(y_true, np.array(prediction) > threshold, beta=2, average='samples')
    return f2, threshold
