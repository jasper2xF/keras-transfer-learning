# -*- coding: utf-8 -*-
"""
Training helper for training and retraining deep learning models.

Training options are training a top model over a pre-loaded model, fine tuning some layers of a pre-loaded model,
retraining a pre-loaded model. A simple CNN model can also be trained from scratch.
"""
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import gc
import logging
import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from kaggle_data.downloader import KaggleDataDownloader
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import fbeta_score
from itertools import chain
from sklearn.model_selection import train_test_split

from keras_transfer_learning import TransferModel
from keras_helper import AmazonKerasClassifier as CNNModel

# Setup logging to std out
logger = logging.getLogger("train_helper")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug("TensorFlow version: " + tf.__version__)


def fine_tune_model(architecture, x_input, y_true, img_size, batch_size, best_fine_weights_path, best_top_weights_path,
                    retrain_epochs_arr, retrain_learn_rates, retrain_momentum_arr, max_train_time_hrs, n_untrained_layers,
                    top_epochs_arr, top_learn_rates, validation_split_size):
    """
    Fine tune a pre-loaded architecture.

    First trains a top model and than retrains the top model together with layers of the pre-loaded architecture.

    :param architecture: Retraining architecture {vgg16, vgg19, resnet50, inceptionv3}
    :param x_input: Training images
    :param y_true: Training labels
    :param img_size: Integer for training image width and height
    :param batch_size: Batch size for training iterations
    :param best_fine_weights_path: Storage path for best fine tuning weights
    :param best_top_weights_path: Storage path for best top model weights
    :param retrain_epochs_arr: Epochs annealing list for fine tuning
    :param retrain_learn_rates: Learning rate annealing list for fine tuning
    :param retrain_momentum_arr: Momentum annealing list for fine tuning
    :param max_train_time_hrs: Maximal training time
    :param n_untrained_layers: Number of layers left untrained for fine tuned model
    :param top_epochs_arr: Epochs annealing list for top model training
    :param top_learn_rates: Learning rate annealing list for top model training
    :param validation_split_size: Double proportion for validation data split
    :return:
    """
    x_train, x_valid, y_train, y_valid = train_test_split(x_input, y_true,
                                                          test_size=validation_split_size)

    # Train top model (so fine tuning partial model is not messed up by large gradients)
    classifier = train_top_model(architecture, img_size, batch_size, best_top_weights_path, top_epochs_arr,
                                 top_learn_rates, x_train, x_valid, y_train, y_valid)

    # Fine-tune full model
    classifier = _train_partial_model(batch_size, best_fine_weights_path, classifier, retrain_epochs_arr,
                                     retrain_learn_rates, retrain_momentum_arr, max_train_time_hrs, n_untrained_layers,
                                     x_train, x_valid, y_train, y_valid)
    # Remove data
    del x_train
    del x_valid
    del y_train
    del y_valid
    gc.collect()
    return classifier


def _train_partial_model(batch_size, best_fine_weights_path, classifier, retrain_epochs_arr,
                        retrain_learn_rates, retrain_momentum_arr, max_train_time_hrs, n_untrained_layers,
                        x_train, x_valid, y_train, y_valid):
    """

    :param batch_size: Batch size for training iterations
    :param best_fine_weights_path: Storage path for best fine tuning weights
    :param classifier: Pre-loaded model with pre-trained top model
    :param retrain_epochs_arr: Epochs annealing list for fine tuning/retraining
    :param retrain_learn_rates: Learning rate annealing list for fine tuning/retraining
    :param retrain_momentum_arr: Momentum annealing list for fine tuning/retraining
    :param max_train_time_hrs: Maximal training time
    :param n_untrained_layers: Number of layers left untrained for fine tuned model (0 for full retraining)
    :param x_train: Training images
    :param x_valid: Validation images
    :param y_train: Training labels
    :param y_valid: Validation labels
    :return:
    """
    # Create a checkpoint, for best model weights
    checkpoint_full = ModelCheckpoint(best_fine_weights_path, monitor='val_loss', verbose=1, save_best_only=True)

    max_train_time_secs = max_train_time_hrs * 60 * 60

    logger.info("Retraining top model and base model layers.")
    logger.info("Will train for max " + str(float(max_train_time_secs) / 60) + " min.")
    classifier.split_fine_tuning_models(n_untrained_layers)
    split_layer_name = classifier.base_model.layers[n_untrained_layers].name
    logger.info("Splitting at: " + split_layer_name + "(last base model layer)")
    classifier.predict_bottleneck_features(x_train, x_valid)

    logger.info("Bottleneck features calculated.")
    train_losses_full, val_losses_full = [], []
    start = time.time()
    first_flag = True
    for learn_rate, epochs, momentum in zip(retrain_learn_rates, retrain_epochs_arr, retrain_momentum_arr):
        if not first_flag:
            #reload best weights from previous run for new annealing run
            classifier.load_top_weights(best_fine_weights_path)
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
    t_epoch = float(end - start) / sum(retrain_epochs_arr)
    logger.info("Training time [min]: " + str(float(end - start) / 60))
    logger.info("Training time [s/epoch]: " + str(t_epoch))
    # Load Best Weights saved by ModelCheckpoint
    classifier.load_top_weights(best_fine_weights_path)

    # Store losses
    np.save("fine_train_losses.npy", train_losses_full)
    np.save("fine_val_losses.npy", val_losses_full)
    # Plot losses
    plt.plot(train_losses_full, label='Training loss')
    plt.plot(val_losses_full, label='Validation loss')
    plt.legend()
    plt.savefig('fine_loss.png')

    # Look at our fbeta_score
    fbeta_score = classifier.get_fbeta_score_valid(y_valid)
    logger.info("Best fine-tuning F2: " + str(fbeta_score))

    return classifier


def retrain_model(architecture, preprocessor, img_size, batch_size, best_retrain_weights_path, best_top_weights_path,
                    retrain_epochs_arr, retrain_learn_rates, retrain_momentum_arr, max_train_time_hrs,
                    top_epochs_arr, top_learn_rates, validation_split_size):
    """
    Retrain a pre-loaded architecture.

    First trains a top model and than retrains the top model together with the pre-loaded architecture.

    :param architecture: Retraining architecture {vgg16, vgg19, resnet50, inceptionv3}
    :param x_input: Training images
    :param y_true: Training labels
    :param img_size: Integer for training image width and height
    :param batch_size: Batch size for training iterations
    :param best_retrain_weights_path: Storage path for best full retrain weights
    :param best_top_weights_path: Storage path for best top model weights
    :param retrain_epochs_arr: Epochs annealing list for retraining
    :param retrain_learn_rates: Learning rate annealing list for retraining
    :param retrain_momentum_arr: Momentum annealing list for retraining
    :param max_train_time_hrs: Maximal training time
    :param top_epochs_arr: Epochs annealing list for top model training
    :param top_learn_rates: Learning rate annealing list for top model training
    :param validation_split_size: Double proportion for validation data split
    :return:
    """

    # Train top model (so retraining model is not messed up by large gradients)
    classifier = train_top_model(architecture, img_size, batch_size, best_top_weights_path, top_epochs_arr,
                                 top_learn_rates, preprocessor)

    # Retrain full model
    classifier = _retrain_full_model(batch_size, best_retrain_weights_path, classifier, retrain_epochs_arr,
                                     retrain_learn_rates, retrain_momentum_arr, max_train_time_hrs,
                                     preprocessor)

    return classifier


def _retrain_full_model(batch_size, best_retrain_weights_path, classifier, retrain_epochs_arr,
                        retrain_learn_rates, retrain_momentum_arr, max_train_time_hrs,
                        preprocessor):
    """

    :param batch_size: Batch size for training iterations
    :param best_fine_weights_path: Storage path for best fine tuning weights
    :param classifier: Pre-loaded model with pre-trained top model
    :param retrain_epochs_arr: Epochs annealing list for fine tuning/retraining
    :param retrain_learn_rates: Learning rate annealing list for fine tuning/retraining
    :param retrain_momentum_arr: Momentum annealing list for fine tuning/retraining
    :param max_train_time_hrs: Maximal training time
    :param x_train: Training images
    :param x_valid: Validation images
    :param y_train: Training labels
    :param y_valid: Validation labels
    :return:
    """
    # Create a checkpoint, for best model weights
    checkpoint_full = ModelCheckpoint(best_retrain_weights_path, monitor='val_loss', verbose=1, save_best_only=True)

    x_train, y_train = preprocessor.X_train, preprocessor.y_train
    x_valid, y_valid = preprocessor.X_val, preprocessor.y_val
    train_generator = preprocessor.get_train_generator(batch_size)
    steps = len(x_train) / batch_size

    max_train_time_secs = max_train_time_hrs * 60 * 60

    logger.info("Retraining top model and base model layers.")
    logger.info("Will train for max " + str(float(max_train_time_secs) / 60) + " min.")
    classifier.set_full_retrain_gen()

    train_losses_full, val_losses_full = [], []
    start = time.time()
    first_flag = True
    for learn_rate, epochs, momentum in zip(retrain_learn_rates, retrain_epochs_arr, retrain_momentum_arr):
        if not first_flag:
            #reload best weights from previous run for new annealing run
            classifier.load_full_weights(best_retrain_weights_path)
        else:
            #first annealing run
            first_flag = False

        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.retrain_full_model_gen(train_generator,
                                                                                          steps,
                                                                                          x_valid,
                                                                                          y_valid,
                                                                                          learn_rate,
                                                                                          momentum,
                                                                                          epochs,
                                                                                          train_callbacks=[checkpoint_full])

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
    t_epoch = float(end - start) / sum(retrain_epochs_arr)
    logger.info("Training time [min]: " + str(float(end - start) / 60))
    logger.info("Training time [s/epoch]: " + str(t_epoch))
    # Load Best Weights saved by ModelCheckpoint
    classifier.load_full_weights(best_retrain_weights_path)
    # Look at our fbeta_score
    fbeta_score = classifier._get_fbeta_score(classifier, x_valid, y_valid)
    logger.info("Best retraining F2: " + str(fbeta_score))

    # Store losses
    np.save("retrain_train_losses.npy", train_losses_full)
    np.save("retrain_val_losses.npy", val_losses_full)
    # Plot losses
    plt.plot(train_losses_full, label='Training loss')
    plt.plot(val_losses_full, label='Validation loss')
    plt.legend()
    plt.savefig('retrain_loss.png')

    return classifier


def train_top_model(architecture, img_size, batch_size, best_top_weights_path, top_epochs_arr, top_learn_rates,
                    preprocessor):
    """
    Train a top model on top of an pre-loaded model architecture.

    :param architecture: Retraining architecture {vgg16, vgg19, resnet50, inceptionv3}
    :param img_size: Integer for training image width and height
    :param batch_size: Batch size for training iterations
    :param best_top_weights_path: Storage path for best top model weights
    :param top_epochs_arr: Epochs annealing list for top model training
    :param top_learn_rates: Learning rate annealing list for top model training
    :param x_train: Training images
    :param x_valid: Validation images
    :param y_train: Training labels
    :param y_valid: Validation labels
    :return:
    """
    # Create a checkpoint, for best model weights
    checkpoint_top = ModelCheckpoint(best_top_weights_path, monitor='val_loss', verbose=1, save_best_only=True)

    x_train, y_train = preprocessor.X_train, preprocessor.y_train
    x_valid, y_valid = preprocessor.X_val, preprocessor.y_val
    train_generator = preprocessor.get_train_generator(batch_size)
    steps = len(x_train) / batch_size

    img_resize = (img_size, img_size)

    # Define and Train model
    n_classes = len(preprocessor.y_map)

    logger.info("Training dense top model.")
    classifier = TransferModel()
    logger.debug("Classifier initialized.")
    classifier.build_base_model(architecture, img_resize, 3)
    logger.info("Base model " + architecture + " built.")
    #classifier.predict_bottleneck_features(x_train, x_valid)
    #logger.debug("Bottleneck features calculated.")
    classifier.build_top_model_gen(n_classes)
    logger.debug("Top built, ready to train.")



    train_losses, val_losses = [], []
    start = time.time()
    for learn_rate, epochs in zip(top_learn_rates, top_epochs_arr):
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_top_model_gen(train_generator,
                                                                                       steps,
                                                                                       x_valid,
                                                                                       y_valid,
                                                                                       learn_rate,
                                                                                       epochs,
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
    # Load Best Weights saved by ModelCheckpoint
    classifier.load_full_weights(best_top_weights_path)
    # Look at our fbeta_score
    fbeta_score = classifier._get_fbeta_score(classifier, x_valid, y_valid)
    logger.info("Best top model F2: " + str(fbeta_score))

    # Store losses
    np.save("top_train_losses.npy", train_losses)
    np.save("top_tval_losses.npy", val_losses)
    # Plot losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.savefig('top_loss.png')

    return classifier


def train_cnn_model(img_size, batch_size, best_cnn_weights_path, cnn_epochs_arr, cnn_learn_rates,
                    validation_split_size, x_train, x_valid, y_train, y_valid):
    """
    Train a CNN model (see keras_helper.py).

    :param img_size: Integer for training image width and height
    :param batch_size: Batch size for training iterations
    :param best_cnn_weights_path: Storage path for best CNN weights
    :param cnn_epochs_arr: Epochs annealing list for retraining
    :param cnn_learn_rates: Learning rate annealing list for retraining
    :param validation_split_size: Double proportion for validation data split
    :param x_train: Training images
    :param x_valid: Validation images
    :param y_train: Training labels
    :param y_valid: Validation labels
    :return:
    """
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
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(x_train, x_valid, y_train, y_valid, learn_rate, epochs,
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
    # Load Best Weights saved by ModelCheckpoint
    classifier.load_weights(best_cnn_weights_path)
    # Look at our fbeta_score
    fbeta_score = classifier._get_fbeta_score(classifier, x_valid, y_valid)
    logger.info("Best CNN model F2: " + str(fbeta_score))

    # Store losses
    np.save("cnn_train_losses.npy", train_losses)
    np.save("cnn_val_losses.npy", val_losses)
    # Plot losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.savefig('cnn_loss.png')

    return classifier


def load_fine_tuned_model(architecture, img_size, n_classes, n_untrained_layers, top_weights_path, fine_weights_path):
    """
    Load a previously fine tuned model via weight npy file.

    Model building requires weights for the original top model. These will be overwritten be the fine tuning weights.

    :param architecture: Retraining architecture {vgg16, vgg19, resnet50, inceptionv3}
    :param img_size: Integer for training image width and height
    :param n_classes: Number of output classes/top model output nodes
    :param n_untrained_layers: Number of layers left untrained for fine tuned model (0 for full retraining)
    :param top_weights_path: Storage path for some top model weights
    :param fine_weights_path: Storage path for fine tuning weights
    :return:
    """
    #TODO: Use top model weights from fine tuning weights for model initialization
    classifier = TransferModel()
    classifier.build_base_model(architecture, [img_size, img_size], 3)
    classifier.build_top_model(n_classes)
    classifier.load_top_weights(top_weights_path)
    classifier.split_fine_tuning_models(n_untrained_layers)
    classifier.load_top_weights(fine_weights_path)
    logger.debug("Loaded " + architecture +" model.")
    return classifier


def load_retrained_model(architecture, img_size, n_classes, weights_path):
    """
    Load a previously retrained model via weight npy file.

    Model building requires weights for the original top model. These will be overwritten be the retraining weights.

    :param architecture: Retraining architecture {vgg16, vgg19, resnet50, inceptionv3}
    :param img_size: Integer for training image width and height
    :param n_classes: Number of output classes/top model output nodes
    :param weights_path: Storage path for full model weights
    :return:
    """
    classifier = TransferModel()
    classifier.build_base_model(architecture, [img_size, img_size], 3)
    classifier.build_top_model_gen(n_classes)
    classifier.set_full_retrain_gen()
    classifier.load_full_weights(weights_path)
    logger.debug("Loaded " + architecture +" model.")
    return classifier


def load_cnn(img_size, n_classes, best_cnn_weights_path):
    """
    Load a previously trained CNN model.

    :param img_size: Integer for training image width and height
    :param n_classes: Number of output classes/top model output nodes
    :param best_cnn_weights_path: Storage path for CNN weights
    :return:
    """
    img_resize = (img_size, img_size)
    classifier = CNNModel()
    classifier.add_conv_layer(img_resize)
    classifier.add_flatten_layer()
    classifier.add_ann_layer(n_classes)
    classifier.load_weights(best_cnn_weights_path)
    logger.debug("Loaded CNN model.")
    return classifier


def create_submission_file(classifiers, preprocessor, batch_size, classification_threshold=0.2):
    """

    :param classifiers: List of classifiers to be used for prediction, can be list of one
    :param x_test: Testing images data
    :param x_test_filename: Testing images file names
    :param y_map: Mapping of text and label id
    :param classification_threshold: Output probability classification threshold (default 0.2)
    :return:
    """
    x_test_filename = preprocessor.X_test
    steps = len(x_test_filename) / batch_size
    y_map = preprocessor.y_map
    predictions = None
    for classifier in classifiers:
        test_gen = preprocessor.get_prediction_generator(batch_size)
        predictions_tmp = classifier.predict_gen(test_gen, steps)
        if predictions is None:
            predictions = predictions_tmp
        else:
            predictions += predictions_tmp

    predictions = predictions / len(classifiers)
    logger.info("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape,
                                                                                           x_test_filename.shape,
                                                                                           predictions[0]))

    thresholds = [classification_threshold] * len(y_map)

    predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)

    # Finally lets assemble and visualize our prediction for the test dataset
    tags_list = [None] * len(predicted_labels)
    for i, tags in enumerate(predicted_labels):
        tags_list[i] = ' '.join(map(str, tags))

    final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]


    final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])

    # And save it to a submission file
    final_df.to_csv('../submission_file.csv', index=False)
    classifier.close()
    return None
