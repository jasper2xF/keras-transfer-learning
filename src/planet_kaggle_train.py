# -*- coding: utf-8 -*-
"""Training script for planet amazon deforestation kaggle."""
import sys
import train_helper

def main():
    run_name = ""

    # Define the dimensions of the image data trained by the network. Due to memory constraints we can't load in the
    # full size 256x256 jpg images. Recommended resized images could be 32x32, 64x64, or 128x128.
    img_size = 64
    #architecture = ["vgg16", "vgg19"]
    architecture = "vgg19"

    # Weights storage parameters
    best_top_weights_path = run_name + "weights_top_best.hdf5"
    best_fine_weights_path = run_name + "weights_full_best.hdf5"
    best_cnn_weights_path = run_name + "weights_cnn_best.hdf5"
    best_retrain_weights_path = run_name + "weights_retrain_best.hdf5"

    #model loading parameters
    n_classes = 17
    load_top_weights_path = "models/weights_top_best_vgg19_x0.hdf5"
    load_full_weights_path = "models/weights_retrain_best_vgg19_x0.hdf5"
    load_cnn_weights_path = "models/weights_cnn_best_0.hdf5"

    # Training parameters
    validation_split_size = 0.2
    batch_size = 128
    # Top model parameters
    top_epochs_arr = [20, 50]
    top_learn_rates = [0.0001, 0.00001]

    # Fine tuning parameters
    max_train_time_hrs = 3
    n_untrained_layers = 0
    fine_epochs_arr = [80]
    fine_learn_rates = [0.0001, 0.00001]
    fine_momentum_arr = [0.9, 0.9]

    # CNN parameters
    cnn_epochs_arr = [20, 50]
    cnn_learn_rates = [0.0001, 0.00001]

    train_top = False
    retrain = True
    train_cnn = False
    load = False
    load_cnn_model = False
    submit = False

    debug = True
    if debug:
        top_epochs_arr = [1, 1]
        fine_epochs_arr = [1, 1]

    train_helper.run(architecture, batch_size, best_cnn_weights_path, best_fine_weights_path, best_retrain_weights_path,
        best_top_weights_path, cnn_epochs_arr, cnn_learn_rates, fine_epochs_arr, fine_learn_rates,
        fine_momentum_arr, submit, img_size, load, load_cnn_model, load_cnn_weights_path,
        load_full_weights_path, load_top_weights_path, max_train_time_hrs, n_classes, n_untrained_layers,
        top_epochs_arr, top_learn_rates, retrain, train_cnn, train_top, validation_split_size)


if __name__ == "__main__":
    main()
