# -*- coding: utf-8 -*-
"""Training script for planet amazon deforestation kaggle."""
import sys
import train_helper

def main():
    run_name = ""

    # Define the dimensions of the image data trained by the network. Due to memory constraints we can't load in the
    # full size 256x256 jpg images. Recommended resized images could be 32x32, 64x64, or 128x128.
    img_size = 64
    architecture = ["vgg16", "vgg19"]
    architecture = "vgg19"

    # Weights parameters
    best_top_weights_path = run_name + "weights_top_best.hdf5"
    best_full_weights_path = run_name + "weights_full_best.hdf5"
    best_cnn_weights_path = run_name + "weights_cnn_best.hdf5"
    best_retrain_weights_path = run_name + "weights_retrain_best.hdf5"
    load_top_weights_path = "models/weights_top_best_vgg19_x0.hdf5"
    load_full_weights_path = "models/weights_retrain_best_vgg19_x0.hdf5"
    load_cnn_weights_path = "models/weights_cnn_best_0.hdf5"

    # Training parameters
    validation_split_size = 0.2
    batch_size = 128
    # Top model parameters
    # learning rate annealing schedule
    top_epochs_arr = [20, 50]
    top_epochs_arr = [1, 1]
    top_learn_rates = [0.0001, 0.00001]
    # top_epochs_arr = [50]
    # top_epochs_arr = [1]
    # top_learn_rates = [0.00001]
    # Fine tuning parameters
    max_train_time_hrs = 3
    n_untrained_layers = 0
    #n_untrained_layers = 0
    #fine_epochs_arr = [5, 50]  # , 300, 500]
    fine_epochs_arr = [80]  # , 300, 500]
    fine_epochs_arr = [1, 1]
    #fine_learn_rates = [0.01, 0.001]  # , 0.0001, 0.00001]
    fine_learn_rates = [0.0001]
    fine_momentum_arr = [0.9, 0.9]  # , 0.9, 0.9]
    annealing = True

    cnn_epochs_arr = [20, 50]
    cnn_learn_rates = [0.0001, 0.00001]
    


    #model loading parameters
    n_classes = 17

    train_top = False
    train = True
    train_cnn = False
    load = False
    load_cnn_model = False
    eval = False
    generate_test = False
    generate_test_ensemble = False

    train_helper.run(annealing, architecture, batch_size, best_cnn_weights_path, best_full_weights_path, best_retrain_weights_path,
        best_top_weights_path, cnn_epochs_arr, cnn_learn_rates, eval, fine_epochs_arr, fine_learn_rates,
        fine_momentum_arr, generate_test, generate_test_ensemble, img_size, load, load_cnn_model, load_cnn_weights_path,
        load_full_weights_path, load_top_weights_path, max_train_time_hrs, n_classes, n_untrained_layers,
        top_epochs_arr, top_learn_rates, train, train_cnn, train_top, validation_split_size)

if __name__ == "__main__":
    main()
