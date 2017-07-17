# -*- coding: utf-8 -*-
"""Training script for planet amazon deforestation kaggle."""
import sys

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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import data_helper
from keras_helper import VGG16DenseRetrainer
from kaggle_data.downloader import KaggleDataDownloader
from keras.callbacks import ModelCheckpoint

import logging
import time

from itertools import chain
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

logger.debug("TensorFlow version: " + tf.__version__)

train_data_dir = "/media/jasper/Data/ml-data/planet_ama_kg/preprocessing/train/"
test_data_dir = "/media/jasper/Data/ml-data/planet_ama_kg/preprocessing/test/"



def get_data():
    # Dataset parameters
    competition_name = "planet-understanding-the-amazon-from-space"

    train, train_u = "train-jpg.tar.7z", "train-jpg.tar"
    test, test_u = "test-jpg.tar.7z", "test-jpg.tar"
    test_additional, test_additional_u = "test-jpg-additional.tar.7z", "test-jpg-additional.tar"
    test_labels = "train_v2.csv.zip"
    destination_path = "../input/"
    is_datasets_present = False

    # If the folders already exists then the files may already be extracted
    # This is a bit hacky but it's sufficient for our needs
    datasets_path = data_helper.get_jpeg_data_files_paths()
    for dir_path in datasets_path:
        if os.path.exists(dir_path):
            is_datasets_present = True

    if not is_datasets_present:
        # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
        downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)

        train_output_path = downloader.download_dataset(train, destination_path)
        downloader.decompress(train_output_path, destination_path)  # Outputs a tar file
        downloader.decompress(destination_path + train_u,
                              destination_path)  # Extract the content of the previous tar file
        os.remove(train_output_path)  # Removes the 7z file
        os.remove(destination_path + train_u)  # Removes the tar file

        test_output_path = downloader.download_dataset(test, destination_path)
        downloader.decompress(test_output_path, destination_path)  # Outputs a tar file
        downloader.decompress(destination_path + test_u,
                              destination_path)  # Extract the content of the previous tar file
        os.remove(test_output_path)  # Removes the 7z file
        os.remove(destination_path + test_u)  # Removes the tar file

        test_add_output_path = downloader.download_dataset(test_additional, destination_path)
        downloader.decompress(test_add_output_path, destination_path)  # Outputs a tar file
        downloader.decompress(destination_path + test_additional_u,
                              destination_path)  # Extract the content of the previous tar file
        os.remove(test_add_output_path)  # Removes the 7z file
        os.remove(destination_path + test_additional_u)  # Removes the tar file

        test_labels_output_path = downloader.download_dataset(test_labels, destination_path)
        downloader.decompress(test_labels_output_path, destination_path)  # Outputs a csv file
        os.remove(test_labels_output_path)  # Removes the zip file
    else:
        logger.info("All datasets are present.")
    return None


def load_train_input(img_size):
    train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
    labels_df = pd.read_csv(train_csv_file)

    # Print all unique tags
    labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
    labels_set = set(labels_list)
    logger.info("There is {} unique labels including {}".format(len(labels_set), labels_set))

    # # Image Resize
    # Define the dimensions of the image data trained by the network. Due to memory constraints we can't load in the full
    # size 256x256 jpg images. Recommended resized images could be 32x32, 64x64, or 128x128.

    img_resize = (img_size, img_size)  # The resize size of each image

    # # Data preprocessing
    # Preprocess the data in order to fit it into the Keras model.
    #
    # Due to the hudge amount of memory the resulting matrices will take, the preprocessing will be splitted into several steps:
    #     - Preprocess training data (images and labels) and train the neural net with it
    #     - Delete the training data and call the gc to free up memory
    #     - Preprocess the first testing set
    #     - Predict the first testing set labels
    #     - Delete the first testing set
    #     - Preprocess the second testing set
    #     - Predict the second testing set labels and append them to the first testing set
    #     - Delete the second testing set

    data_dir = os.path.join(train_data_dir, str(img_size))
    if os.path.exists(data_dir):
        x_input, y_input, y_map = data_helper.load_data(data_dir)
    else:
        x_input, y_input, y_map = data_helper.preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
        data_helper.store_data(data_dir, x_input, y_input, y_map)
        # Free up all available memory space after this heavy operation
        gc.collect();

    logger.debug("x_input shape: {}".format(x_input.shape))
    logger.debug("y_input shape: {}".format(y_input.shape))
    logger.debug("Label mapping: " + str(y_map))
    return x_input, y_input, y_map


def load_test_input(img_size):
    train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
    img_resize = (img_size, img_size)

    data_dir = os.path.join(test_data_dir, str(img_size))
    if os.path.exists(data_dir):
        x_test, x_test_filename = data_helper.load_test_data(data_dir)
        return x_test, x_test_filename

    x_test_org, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)
    # Predict the labels of our x_test images
    del x_test
    x_test_additional, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)
    del x_test
    x_test = np.vstack((x_test_org, x_test_additional))
    del x_test_org
    del x_test_additional
    gc.collect()
    x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
    logger.info("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape,
                                                                                           x_test_filename.shape,
                                                                                           predictions[0]))
    data_helper.save_test_data(x_test, x_test_filename)
    return x_test, x_test_filename


def fine_tune_vgg16(x_input, y_input, y_map, img_size):

    # Weights parameters
    best_top_weights_path = "weights_top_best.hdf5"
    best_full_weights_path = "weights_full_best.hdf5"


    # Training parameters
    validation_split_size = 0.2
    batch_size = 128

    # Top model parameters
    # learning rate annealing schedule
    #top_epochs_arr = [10, 5, 5]
    #top_learn_rates = [0.001, 0.0001, 0.00001]
    top_epochs_arr = [50]
    #top_epochs_arr = [1]
    top_learn_rates = [0.00001]

    # Fine tuning parameters
    max_train_time_hrs = 3
    n_untrained_layers = 10

    fine_epochs_arr = [5, 50]#, 300, 500]
    #fine_epochs_arr = [1, 1, 1, 1]
    fine_learn_rates = [0.01, 0.001]#, 0.0001, 0.00001]
    fine_momentum_arr = [0.9, 0.9]#, 0.9, 0.9]
    annealing = True

    # Create a checkpoint, for best model weights
    checkpoint_top = ModelCheckpoint(best_top_weights_path, monitor='val_acc', verbose=1, save_best_only=True)
    checkpoint_full = ModelCheckpoint(best_full_weights_path, monitor='val_acc', verbose=1, save_best_only=True)

    classifier = train_top_model(img_size, batch_size, best_top_weights_path, checkpoint_top, top_epochs_arr, top_learn_rates,
                                 validation_split_size, x_input, y_input)

    # ## Fine-tune full model
    #
    # Now we retrain the top model together with the last convolutional layer from the VGG16 model
    classifier = train_partial_vgg16(batch_size, best_full_weights_path, checkpoint_full, classifier, fine_epochs_arr,
                        fine_learn_rates, fine_momentum_arr, max_train_time_hrs, n_untrained_layers,
                        validation_split_size, x_input, y_input, annealing)

    return classifier


def train_partial_vgg16(batch_size, best_full_weights_path, checkpoint_full, classifier, fine_epochs_arr,
                        fine_learn_rates, fine_momentum_arr, max_train_time_hrs, n_untrained_layers,
                        validation_split_size, x_input, y_input, annealing):
    max_train_time_secs = max_train_time_hrs * 60 * 60
    X_train, X_valid, y_train, y_valid = train_test_split(x_input, y_input,
                                                          test_size=validation_split_size)
    logger.info("Fine tuning top model and VGG16 layers.")
    logger.info("Will train for max " + str(float(max_train_time_secs) / 60) + " min.")
    init_top_weights = classifier.split_fine_tuning_models(n_untrained_layers)
    split_layer_name = classifier.base_model.layers[n_untrained_layers].name
    logger.info("Splitting at: " + split_layer_name)
    classifier.predict_bottleneck_features(X_train, X_valid, validation_split_size=validation_split_size)
    del X_train
    del X_valid
    gc.collect()
    logger.info("Bottleneck features calculated.")
    train_losses_full, val_losses_full = [], []
    start = time.time()
    for learn_rate, epochs, momentum in zip(fine_learn_rates, fine_epochs_arr, fine_momentum_arr):
        if not annealing:
            classifier.set_top_weights(init_top_weights)
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.fine_tune_full_model(y_train, y_valid, learn_rate,
                                                                                        momentum, epochs,
                                                                                        batch_size,
                                                                                        validation_split_size,
                                                                                        train_callbacks=[
                                                                                            checkpoint_full])

        train_losses_full += tmp_train_losses
        val_losses_full += tmp_val_losses
        logger.info("learn_rate : " + str(learn_rate))
        logger.info("epochs : " + str(epochs))
        logger.info("momentum : " + str(momentum))
        logger.info("fbeta_score : " + str(fbeta_score))

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
    np.save("fine_tval_losses.npy", val_losses_full)
    # Look at our fbeta_score
    fbeta_score = classifier.get_fbeta_score_valid(y_valid)
    logger.info("Best fine-tuning F2: " + str(fbeta_score))
    return classifier


def train_top_model(img_size, batch_size, best_top_weights_path, checkpoint_top, top_epochs_arr, top_learn_rates,
                    validation_split_size, x_input, y_input):
    img_resize = (img_size, img_size)

    # Define and Train model
    n_classes = y_input.shape[1]
    X_train, X_valid, y_train, y_valid = train_test_split(x_input, y_input,
                                                          test_size=validation_split_size)
    logger.info("Training dense top model.")
    classifier = VGG16DenseRetrainer()
    logger.info("Classifier initialized.")
    classifier.build_vgg16(img_resize, 3, n_classes)
    logger.info("Vgg16 built.")
    classifier.predict_bottleneck_features(X_train, X_valid, validation_split_size=validation_split_size)
    del X_train
    del X_valid
    gc.collect()
    logger.info("Vgg16 bottleneck features calculated.")
    classifier.build_top_model(n_classes)
    logger.info("Top built, ready to train.")
    train_losses, val_losses = [], []
    start = time.time()
    for learn_rate, epochs in zip(top_learn_rates, top_epochs_arr):
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_top_model(y_train, y_valid, learn_rate, epochs,
                                                                                   batch_size,
                                                                                   validation_split_size=validation_split_size,
                                                                                   train_callbacks=[checkpoint_top])
        train_losses += tmp_train_losses
        val_losses += tmp_val_losses

        logger.info("learn_rate : " + str(learn_rate))
        logger.info("epochs : " + str(epochs))
        logger.info("fbeta_score : " + str(fbeta_score))
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
    # Look at our fbeta_score
    fbeta_score = classifier.get_fbeta_score_valid(y_valid)
    logger.info("Best top model F2: " + str(fbeta_score))
    return classifier


def create_test_file(classifier, x_test, x_test_filename , y_map, classification_threshold=0.2):

    predictions = classifier.predict(x_test)

    # Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class.
    #
    # To do so we will take the median value of each classes.

    # For now we'll just put all thresholds to 0.2
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


if __name__ == "__main__":

    img_size = 64
    """
    get_data()
    x_input, y_input, y_map = load_train_input(img_size)
    classifier = fine_tune_vgg16(x_input, y_input, y_map, img_size)
    del x_input
    del y_input
    gc.collect()
    """
    x_test, x_test_filename = load_test_input(img_size)
    #create_test_file(classifier, x_test, x_test_filename, y_map)
