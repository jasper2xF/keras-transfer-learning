import numpy as np
import os
import h5py

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras import backend
from keras.engine import topology
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import optimizers

class LossHistory(Callback):
    """
    Class for loss history through epoch end callback.

    on_epoch_end(self, epoch, logs={})
    """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        """Append 'loss' and 'val_loss' to internal data structure."""
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        return None


class TransferModel:
    """
    Class for retraining VGG16, VGG19, ResNet-50, Inception-v3 architectures.

    Implementation allows generator input for training on large data sets under memory constraints. Non generator
    implementation available for top model training and fine tuning.

    Supports retraining dense top_model:
        classifier = TransferModel()
        classifier.build_base_model(architecture, img_resize, 3)
        classifier.add_top_model(n_classes)
        classifier.train_top_model_gen(train_generator, steps, x_valid, y_valid, learn_rate=0.001, ...)

    Supports retraining dense top_model with layers of base architecture. Note that this requires training vanilla
    dense top_model first:
        *dense top model steps*
        classifier.set_base_model_fine_tuning(n_untrained_layers)
        classifier.train_top_model_gen(train_generator, steps, x_valid, y_valid, learn_rate=0.001, ...)

    Supports retraining dense top_model with full base architecture. Note that this requires training vanilla dense
    top_model first:
        *dense top model steps*
        classifier.retrain_full_model_gen(train_generator, steps, x_valid, y_valid, learn_rate=0.001, ...)

    Supports efficient top model training or fine tuning with one time bottleneck feature calculation by keeping base
    model separate.
        classifier = TransferModel(separate_top_model=False)
        classifier.build_base_model(architecture, img_resize, 3)
        classifier.add_top_model(n_classes)
        bottleneck_feat_trn = classifier.predict_bottleneck_features_gen(train_generator, steps)
        bottleneck_feat_val = classifier.predict_bottleneck_features(x_valid)
        classifier.train_top_model(bottleneck_feat_trn, y_train, bottleneck_feat_val, y_valid, learn_rate=0.001, ...)
    """
    def __init__(self, separate_top_model=False):
        """

        :param separate_top_model: boolean
            Separates base and top model if set, allowing efficient fine tuning on bottleneck features
        """
        self.losses = []
        self.model = None
        self.classification_threshold = 0.2
        self.n_base_model_layers = None
        self.separate_top_model = separate_top_model
        self.top_model = None

    def build_base_model(self, architecture, img_size, img_channels):
        """
        Set base model to pre-trained architecture.

        Options are:
        vgg16
        vgg19
        resnet50
        inceptionv3

        :param architecture: string
            Architecture {vgg16, vgg19, resnet50, inceptionv3}
        :param img_size: 2d tuple <int>
            Image width x height
        :param img_channels: int
            Number of image channels
        :return: none
        """

        if architecture == "vgg16":
            self.build_vgg16(img_size, img_channels)
        elif architecture == "vgg19":
            self.build_vgg19(img_size, img_channels)
        elif architecture == "resnet50":
            self.build_resnet50(img_size, img_channels)
        elif architecture == "inceptionv3":
            self.build_inceptionv3(img_size, img_channels)
        else:
            raise ValueError("Invalid architecture: '" + architecture + "'")
        # Store number of base model layers
        self.n_base_model_layers = len(self.model.layers)
        return None

    def build_vgg16(self, img_size, img_channels):
        """
        VGG16 model, with weights pre-trained on ImageNet.

        Width and height should be no smaller than 48.

        :param img_size: 2d tuple <int>
            Image width x height
        :param img_channels: int
            Number of image channels
        :return: none
        """
        img_width = img_size[0]
        img_height = img_size[1]

        self.model = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(img_width, img_height, img_channels),
            pooling=None
        )
        return None

    def build_vgg19(self, img_size, img_channels):
        """
        VGG19 model, with weights pre-trained on ImageNet.

        Width and height should be no smaller than 48.

        :param img_size: 2d tuple <int>
            Image width x height
        :param img_channels: int
            Number of image channels
        :return: none
        """
        img_width = img_size[0]
        img_height = img_size[1]

        self.model = VGG19(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(img_width, img_height, img_channels),
            pooling=None
        )
        return None

    def build_resnet50(self, img_size, img_channels):
        """
        ResNet50 model, with weights pre-trained on ImageNet.

        Width and height should be no smaller than 197.

        :param img_size: 2d tuple <int>
            Image width x height
        :param img_channels: int
            Number of image channels
        :return: none
        """
        img_width = img_size[0]
        img_height = img_size[1]
        #width and height should be no smaller than 71
        self.model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(img_width, img_height, img_channels),
            pooling=None
        )
        return None

    def build_inceptionv3(self, img_size, img_channels):
        """
        Inception V3 model, with weights pre-trained on ImageNet.

        Width and height should be no smaller than 139

        :param img_size: 2d tuple <int>
            Image width x height
        :param img_channels: int
            Number of image channels
        :return: none
        """
        img_width = img_size[0]
        img_height = img_size[1]
        # width and height should be no smaller than 71
        self.model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(img_width, img_height, img_channels),
            pooling=None)
        return None

    def add_top_model(self, n_classes, n_dense=256, dropout_rate=0.3):
        """
        Build a top model top use on top of base architecture.

        Creates one dense layer of 'relu' units of dimension n_dense and an output layer of 'sigmoid' units or
        dimension n_classe.
        :param n_classes: int
            Number of output 'sigmoid' units
        :param n_dense: int
            Number of dense layer 'relu' units (default=256)
        :param dropout_rate: float
            Dropout rate for dense layer (default=0.3)
        :return: None
        """
        if self.model is None:
            raise ValueError("Current model is empty, please build a base model.")

        top_model = Sequential()
        top_model.add(Flatten(input_shape=self.model.output_shape[1:]))
        top_model.add(Dense(n_dense, activation='relu'))
        top_model.add(Dropout(dropout_rate))
        top_model.add(Dense(n_classes, activation='sigmoid'))

        if self.separate_top_model:
            # Keep top model separate
            self.top_model = top_model
        else:
            # Update model to new top model on top of base model
            self.model = Model(input=self.model.input, output=top_model(self.model.output))
        return None

    def set_base_model_fine_tuning(self, layer_id):
        """
        Deactivates retraining of base model layers up to layer_id (inclusive). Enables retraining of remaining layers.

        :param layer_id: int
            Id of last base model layer deactivated for training.
        :return: None
        """
        if layer_id > self.n_base_model_layers:
            raise ValueError("Layer id {0:d} > number of base model layers {1:d}"
                             .format(layer_id, self.n_base_model_layers))

        if self.separate_top_model:
            self._split_fine_tuning_models(layer_id)
        else:
            self._set_fine_tuning_layers(layer_id)
        return None

    def _split_fine_tuning_models(self, split_layer_id):
        """
        Splits model at split_layer_id into base model and new top model.

        Original model consists of previous base and top model. New base goes until split_layer_id (inclusive). New
        top model starts at split_layer_id (exclusive).

        :param split_layer_id: int
            Id of last base model layer deactivated for training.
        :return: None
        """
        # Create full model for splitting
        full_model = Model(input=self.model.input, output=self.top_model(self.model.output))

        # Create new base model from input to split layer
        self.model = Model(input=full_model.input, output=full_model.layers[split_layer_id].output)

        # Create new top model from base model output to previous top model end
        top_model_input = Input(shape=self.model.output_shape[1:])
        x = top_model_input
        for layer in full_model.layers[(split_layer_id + 1):]:
            x = layer(x)
        self.top_model = Model(top_model_input, x)

        return None

    def _set_fine_tuning_layers(self, layer_id):
        """

        :param layer_id:
        :return:
        """
        # Increase id, so deactivation is inclusive
        layer_id += 1
        for layer in self.model.layers[:layer_id]:
            layer.trainable = False
        for layer in self.model.layers[layer_id:]:
            layer.trainable = True
        return False

    def disable_base_model_training(self):
        """
        Set retraining of base model layers to false

        :return: None
        """
        for layer in self.model.layers[:self.n_base_model_layers]:
            layer.trainable = False
        return None

    def enable_base_model_training(self):
        """
        Sets base model layers to trainable in full model.

        :return: Weights of model
        """
        for layer in self.model.layers[:self.n_base_model_layers]:
            layer.trainable = True
        return None

    def predict_bottleneck_features(self, x_input):
        """
        Only available when using separate top model. Predicts output of base model.

        :param x_input: np array
            Image data input
        :return: np array
            Bottleneck feature prediction
        """
        self._seperate_model_check(True)

        bottleneck_feat= self.model.predict(x_input)
        return bottleneck_feat

    def predict_bottleneck_features_gen(self, input_gen, steps):
        """
        Only available when using separate top model. Predicts output of base model based on generator.

        :param input_gen:
        :param steps:
        :return: np array
            Bottleneck feature prediction
        """
        self._seperate_model_check(True)

        bottleneck_feat= self.model.predict_generator(
            generator=input_gen,
            verbose=2,
            steps=steps
        )
        return bottleneck_feat

    def train_gen(self, epochs, learn_rate, steps, train_callbacks, train_generator, x_valid, y_valid,
                  early_stop_patience=5):
        """
        Train current model. Make sure to enable/disable base model training or set fine tuning beforehand.

        :param train_generator: generator
            Yields training images and labels
        :param steps: int
            Number of batch steps in generator
        :param x_valid: Numpy array
            Array of training data
        :param y_valid: Numpy array
            Array of labels
        :param learn_rate: float
            Learning rate for SDG
        :param epochs:
            Number of training epochs
        :param train_callbacks: keras callback
            Training callbacks for fit method
        :param early_stop_patience: int
            Number of no loss improvement steps before early stopping
        :return: [Training losses, validation losses, f2 score]
        """
        self._seperate_model_check(False)

        history = LossHistory()

        opt = Adam(lr=learn_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=0, mode='auto')

        self.model.fit_generator(train_generator,
                                 steps,
                                 epochs=epochs,
                                 verbose=2,
                                 validation_data=(x_valid, y_valid),
                                 callbacks=[history, *train_callbacks, early_stopping])

        # determine classification threshold
        self.fit_classification_threshold(self.model, x_valid, y_valid)
        # Eval on validation data
        fbeta_score = self._get_fbeta_score(self.model, x_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def train_top_model_gen(self, train_generator, steps, x_valid, y_valid, learn_rate=0.001, epoch=5,
                            train_callbacks=()):
        """
        Train top model as part of full model architecture.

        :param train_generator: generator
            Yields training images and labels
        :param steps: int
            Number of batch steps in generator
        :param x_valid: Numpy array
            Array of training data
        :param y_valid: Numpy array
            Array of labels
        :param learn_rate: float
            Learning rate for SDG
        :param epoch:
            Number of training epochs
        :param train_callbacks: keras callback
            Training callbacks for fit method
        :return: [Training losses, validation losses, f2 score]
        """
        self._seperate_model_check(False)

        # Set retraining of base model layers to false
        self.disable_base_model_training()

        train_losses, val_losses, fbeta_score = self.train_gen(epoch, learn_rate, steps, train_callbacks,
                                                               train_generator, x_valid, y_valid, early_stop_patience)

        return [train_losses, val_losses, fbeta_score]

    def retrain_full_model_gen(self, train_generator, steps, x_valid, y_valid, learn_rate=0.001, momentum=0.9, epoch=5,
                            train_callbacks=(), early_stop_patience=5):
        """
        Retrain full model via generator.

        :param train_generator: generator
            Yields training images and labels
        :param steps: int
            Number of batch steps in generator
        :param x_valid: Numpy array
            Array of training data
        :param y_valid: Numpy array
            Array of labels
        :param learn_rate: float
            Learning rate for SDG
        :param momentum: float
            Momentum for SGD
        :param epoch:
            Number of training epochs
        :param train_callbacks: keras callback
            Training callbacks for fit method
        :param early_stop_patience: int
            Number of no loss improvement steps before early stopping
        :return: [Training losses, validation losses, f2 score]
        """
        self._seperate_model_check(False)

        # Ensure base model is trainable, shouldn't be necessary
        self.enable_base_model_training()

        train_losses, val_losses, fbeta_score = self.train_gen(epoch, learn_rate, steps, train_callbacks,
                                                               train_generator, x_valid, y_valid, early_stop_patience)

        return [train_losses, val_losses, fbeta_score]

    def fine_tune_gen(self, last_base_layer, train_generator, steps, x_valid, y_valid, learn_rate=0.001, momentum=0.9,
                      epoch=5, train_callbacks=(), early_stop_patience=10):
        """
        Fine tune partial model after last_base_layer (exclusive) via generator.

        :param last_base_layer: int
            Last base model layer (inclusive) that will not be trained
        :param train_generator: generator
            Yields training images and labels
        :param steps: int
            Number of batch steps in generator
        :param x_valid: Numpy array
            Array of training data
        :param y_valid: Numpy array
            Array of labels
        :param learn_rate: float
            Learning rate for SDG
        :param momentum: float
            Momentum for SGD
        :param epoch:
            Number of training epochs
        :param train_callbacks: keras callback
            Training callbacks for fit method
        :param early_stop_patience: int
            Number of no loss improvement steps before early stopping
        :return: [Training losses, validation losses, f2 score]
        """
        self._seperate_model_check(False)

        self.set_base_model_fine_tuning(last_base_layer)

        train_losses, val_losses, fbeta_score = self.train_gen(epoch, learn_rate, steps, train_callbacks,
                                                               train_generator, x_valid, y_valid, early_stop_patience)

        return [train_losses, val_losses, fbeta_score]

    def train_top_model(self, bottleneck_feat_trn, y_train, bottleneck_feat_val, y_valid, learn_rate=0.001, epoch=5,
                        batch_size=128, train_callbacks=()):
        """
        Train top model on bottleneck features.

        With cross entropy loss and adam optimizer.
        """
        self._seperate_model_check(True)

        history = LossHistory()

        opt = Adam(lr=learn_rate)

        self.top_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.top_model.fit(bottleneck_feat_trn, y_train,
                           epochs=epoch,
                           batch_size=batch_size,
                           verbose=2,
                           validation_data=(bottleneck_feat_val, y_valid),
                           callbacks=[history, *train_callbacks, early_stopping])
        #determine classification threshold
        self.fit_classification_threshold(self.top_model, bottleneck_feat_trn, y_train)
        fbeta_score = self._get_fbeta_score(self.top_model, bottleneck_feat_val, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def fine_tune_full_model(self, bottleneck_feat_trn, y_train, bottleneck_feat_val, y_valid, learn_rate=0.001,
                             momentum=0.9, epoch=5, batch_size=128, train_callbacks=(), early_stop_patience=10):
        """
        DEPRECATED IN FAVOR OF GENERATOR METHOD
        Retrain top model with layers of base model.

        Uses binary cross entropy loss, SGD optimzer with momentum and early stopping.
        """
        self._seperate_model_check(True)

        history = LossHistory()

        self.top_model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=learn_rate, momentum=momentum),
                      metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=0, mode='auto')
        self.top_model.fit(bottleneck_feat_trn, y_train,
                           epochs=epoch,
                           batch_size=batch_size,
                           verbose=2,
                           validation_data=(bottleneck_feat_val, y_valid),
                           callbacks=[history, *train_callbacks, early_stopping])
        self.fit_classification_threshold(self.top_model, bottleneck_feat_trn, y_train)
        fbeta_score = self._get_fbeta_score(self.top_model, bottleneck_feat_val, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_top_weights(self, weight_file_path):
        """
        Save top model weights.
        """
        self.top_model.save_weights(weight_file_path)
        return None

    def load_top_weights(self, weight_file_path):
        """
        Load top model weights.
        """
        self.top_model.load_weights(weight_file_path)
        return None

    def get_top_weights(self):
        """
        Set top model weights.
        """
        return self.top_model.get_weights()

    def load_weights(self, weight_file_path):
        """
        Load model weights.

        Refers to base model if separate top model.
        """
        self.model.load_weights(weight_file_path)
        return None

    def set_weights(self, weights):
        """
        Set model weights.

        Refers to base model if separate top model.
        """
        self.model.set_weights(weights)
        return None

    def get_weights(self, weights):
        """
        Get  model weights.

        Refers to base model if separate top model.
        """
        return self.model.get_weights(weights)

    def predict(self, x_input):
        """
        Predict output for given input.

        :param x_input: Model input
        :return: Predictions of output layer
        """
        if self.separate_top_model:
            bottleneck_feat = self.model.predict(x_input)
            predictions= self.top_model.predict(bottleneck_feat)
        else:
            predictions = self.model.predict(x_input)
        return predictions

    def predict_gen(self, test_gen, steps):
        """
        Predict output for given input.

        :param test_gen: generator
            Yields test images
        :param steps: int
            Number of batch steps in generator
        :return: Predictions of output layer
        """
        if self.separate_top_model:
            bottleneck_feat = self.model.predict_generator(
                generator=test_gen,
                verbose=2,
                steps=steps
            )
            predictions= self.top_model.predict(bottleneck_feat)
        else:
            predictions = self.model.predict_generator(
                generator=test_gen,
                verbose=2,
                steps=steps
            )
        return predictions

    def _seperate_model_check(self, expected_val):
        """

        :param expected_val:
        :return:
        """
        if expected_val:
            if not self.separate_top_model:
                raise ValueError("Method can only be used when using separate top model.")
        else:
            if not self.separate_top_model:
                raise ValueError("Method can only be used when not using separate top model.")
        return None

    def _get_fbeta_score(self, classifier, x_valid, y_valid, threshold=None):
        """DEPRECATED - Calculate F2 score."""
        # TODO: Move out, class is not responsible for evaluation
        if threshold is None:
            threshold = self.classification_threshold
        p_valid = classifier.predict(x_valid)
        return fbeta_score(y_valid, np.array(p_valid) > threshold, beta=2, average='samples')

    def fit_classification_threshold(self, classifier, x_input, y_true, t_max=5):
        """
        DEPRECATED - Fits classification threshold to maximize f2 score.

        Threshold ranges from 0.1 to 0.t_max with 0.1 steps.

        :param classifier: classifier to be used for prediction.
        :param x_input: input data for fitting
        :param y_true: gold labels for fitting
        :param t_max: integer for max threshold 0.t_max
        :return:
        """
        # TODO: Move out, class is not responsible
        prediction = classifier.predict(x_input)

        best_f2 = 0
        for i in range(1,t_max+1):
            threshold = float(i)/10
            f2 = fbeta_score(y_true, np.array(prediction) > threshold, beta=2, average='samples')
            if f2 > best_f2:
                best_f2 = f2
                self.classification_threshold = threshold

        return self.classification_threshold

    def map_predictions(self, predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        """Clears backend session."""
        backend.clear_session()
        return None