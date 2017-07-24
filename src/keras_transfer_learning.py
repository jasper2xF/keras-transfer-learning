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


class TransferModel:
    """
    Class for retraining VGG16 architecture.

    Supports retraining dense top_model:
        classifier = TransferModel()
        classifier.build_base_model(architecture, img_size, img_channels)
        classifier.build_top_model(n_classes, n_dense=256, dropout_rate=0.3)
        classifier.train_top_model(y_train, y_valid, learn_rate=0.001, epoch=5, batch_size=128, train_callbacks=())
    Supports retraining dense top_model with layers of base architecture. Note that this requires training vanilla
    dense top_model first:
        *dense top model steps*
        classifier.split_fine_tuning_models(split_layer_id)
        classifier.predict_bottleneck_features(x_train, x_valid)
        classifier.fine_tune_full_model(y_train, y_valid, learn_rate=0.001, epoch=5, batch_size=128, train_callbacks=())
    Supports retraining dense top_model with full base architecture. Note that this requires training vanilla dense
    top_model first:
        *dense top model steps*
        classifier.set_full_retrain()
        classifier.retrain_full_model(y_train, y_valid, learn_rate=0.001, epoch=5, batch_size=128, train_callbacks=())
    """
    def __init__(self):
        self.losses = []
        self.base_model = None
        self.top_model = Sequential()
        self.full_model = None
        self.bottleneck_feat_trn = None
        self.bottleneck_feat_val = None
        self.classification_threshold = 0.2

    def build_base_model(self, architecture, img_size, img_channels):
        """
        Set base model to pre-trained architecture.

        Options are:
        vgg16
        vgg19
        resnet50
        inceptionv3
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
        return None

    def build_vgg16(self, img_size, img_channels):
        """
        VGG16 model, with weights pre-trained on ImageNet.

        Width and height should be no smaller than 48.
        """
        img_width = img_size[0]
        img_height = img_size[1]

        self.base_model = VGG16(include_top=False, weights='imagenet',
              input_tensor=None, input_shape=(img_width, img_height, img_channels),
              pooling=None)

    def build_vgg19(self, img_size, img_channels):
        """
        VGG19 model, with weights pre-trained on ImageNet.

        Width and height should be no smaller than 48.
        """
        img_width = img_size[0]
        img_height = img_size[1]

        self.base_model = VGG19(include_top=False, weights='imagenet',
              input_tensor=None, input_shape=(img_width, img_height, img_channels),
              pooling=None)

    def build_resnet50(self, img_size, img_channels):
        """
        ResNet50 model, with weights pre-trained on ImageNet.

        Width and height should be no smaller than 197.
        """
        img_width = img_size[0]
        img_height = img_size[1]
        #width and height should be no smaller than 71
        self.base_model = ResNet50(include_top=False, weights='imagenet',
                                input_tensor=None, input_shape=(img_width, img_height, img_channels),
                                pooling=None)

    def build_inceptionv3(self, img_size, img_channels):
        """
        Inception V3 model, with weights pre-trained on ImageNet.

        Width and height should be no smaller than 139
        """
        img_width = img_size[0]
        img_height = img_size[1]
        # width and height should be no smaller than 71
        self.base_model = InceptionV3(include_top=False, weights='imagenet',
                                   input_tensor=None, input_shape=(img_width, img_height, img_channels),
                                   pooling=None)

    def predict_bottleneck_features(self, x_train, x_valid):
        """
        DEPRECATED - Runs input through base model to predict, store and return bottleneck features.

        Warning: Bottleneck features are stored as member variables. This can cause memory usage issues.
        """
        # TODO: Do not store bottleneck features internally but return and let user handle these large variables
        self.bottleneck_feat_trn = self.base_model.predict(x_train)
        self.bottleneck_feat_val = self.base_model.predict(x_valid)
        return self.bottleneck_feat_trn, self.bottleneck_feat_val

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

    def get_fbeta_score_valid(self, y_valid):
        """
        DEPRECATED - Calculate F2 score based on gold labels and stored bottleneck features.

        :param y_valid:
        :return:
        """
        # TODO: Move out, class is not responsible for evaluation
        fbeta_score = self._get_fbeta_score(self.top_model, self.bottleneck_feat_val, y_valid)
        return fbeta_score

    def build_top_model(self, n_classes, n_dense=256, dropout_rate=0.3):
        """
        Build a top model top use on top of base architecture.

        Creates one dense layer of 'relu' units of dimension n_dense and an output layer of 'sigmoid' units or
        dimension n_classe.
        :param n_classes: Number of output 'sigmoid' units
        :param n_dense: Number of dense layer 'relu' units (default=256)
        :param dropout_rate: Dropout rate for dense layer (default=0.3)
        :return:
        """
        self.top_model.add(Flatten(input_shape=self.base_model.output_shape[1:]))
        self.top_model.add(Dense(n_dense, activation='relu'))
        self.top_model.add(Dropout(dropout_rate))
        self.top_model.add(Dense(n_classes, activation='sigmoid'))

    def train_top_model(self, y_train, y_valid, learn_rate=0.001, epoch=5, batch_size=128, train_callbacks=()):
        """Train top model with cross entropy loss and adam optimizer."""
        history = LossHistory()

        opt = Adam(lr=learn_rate)

        self.top_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.top_model.fit(self.bottleneck_feat_trn, y_train,
                           epochs=epoch,
                           batch_size=batch_size,
                           verbose=2,
                           validation_data=(self.bottleneck_feat_val, y_valid),
                           callbacks=[history, *train_callbacks, early_stopping])
        #determine classification threshold
        self.fit_classification_threshold(self.top_model, self.bottleneck_feat_trn, y_train)
        fbeta_score = self._get_fbeta_score(self.top_model, self.bottleneck_feat_val, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_top_weights(self, weight_file_path):
        """
        Save top model weights.

        Applies to all weights if set_full_retrain() was called.
        """
        self.top_model.save_weights(weight_file_path)

    def load_top_weights(self, weight_file_path):
        """
        Lead top model weights.

        Applies to all weights if set_full_retrain() was called.
        """
        self.top_model.load_weights(weight_file_path)

    def set_top_weights(self, weights):
        """
        Set top model weights.

        Applies to all weights if set_full_retrain() was called.
        """
        self.top_model.set_weights(weights)

    def get_top_weights(self):
        """
        Set top model weights.

        Applies to all weights if set_full_retrain() was called.
        """
        return self.top_model.get_weights()

    def split_fine_tuning_models(self, split_layer_id):
        """
        Splits full model at split_layer_id and returns weights of new top model.

        Full model consists of previous base_model and top_model. New base_model goes until split_layer_id (inclusive).
        New top_model starts at split_layer_id (exclusive). The top_model weights are stored and returned as starting
        point for future retraining.
        """
        # Create full model for splitting
        full_model = Model(input=self.base_model.input, output=self.top_model(self.base_model.output))

        # Create new base model from input to split layer
        self.base_model = Model(input=full_model.input, output=full_model.layers[split_layer_id].output)

        # Create new top model from base model output to previous top model end
        top_model_input = Input(shape=self.base_model.output_shape[1:])
        x = top_model_input
        for layer in full_model.layers[(split_layer_id + 1):]:
            x = layer(x)
        self.top_model = Model(top_model_input, x)

        return self.top_model.get_weights()

    def set_full_retrain(self):
        """
        Combines base and top model into single trainable architecture.

        From now on top model weights methods apply to full model.
        :return: Weights of model
        """

        self.top_model = Model(input=self.base_model.input, output=self.top_model(self.base_model.output))
        self.base_model = None
        return self.top_model.get_weights()

    def fine_tune_full_model(self, y_train, y_valid, learn_rate=0.001, momentum=0.9, epoch=5, batch_size=128,
                             train_callbacks=(), early_stop_patience=10):
        """
        Retrain top model with layers of base model.

        Uses binary cross entropy loss, SGD optimzer with momentum and early stopping.
        """
        # TODO: Do not store bottleneck features internally but let user pass them

        history = LossHistory()

        self.top_model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=learn_rate, momentum=momentum),
                      metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=0, mode='auto')
        self.top_model.fit(self.bottleneck_feat_trn, y_train,
                           epochs=epoch,
                           batch_size=batch_size,
                           verbose=2,
                           validation_data=(self.bottleneck_feat_val, y_valid),
                           callbacks=[history, *train_callbacks, early_stopping])
        self.fit_classification_threshold(self.top_model, self.bottleneck_feat_trn, y_train)
        fbeta_score = self._get_fbeta_score(self.top_model, self.bottleneck_feat_val, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def retrain_full_model(self, X_train, X_valid, y_train, y_valid, learn_rate=0.001, momentum=0.9, epoch=5, batch_size=128,
                             train_callbacks=(), early_stop_patience=10):
        """
        Retrain top model with full base model.

        Uses binary cross entropy loss, SGD optimzer with momentum and early stopping.
        """
        # TODO: Do not store bottleneck features internally but let user pass them

        history = LossHistory()

        self.top_model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=learn_rate, momentum=momentum),
                      metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=0, mode='auto')
        self.top_model.fit(X_train, y_train,
                           epochs=epoch,
                           batch_size=batch_size,
                           verbose=2,
                           validation_data=(X_valid, y_valid),
                           callbacks=[history, *train_callbacks, earlyStopping])
        self.fit_classification_threshold(self.top_model, X_train, y_train)
        fbeta_score = self._get_fbeta_score(self.top_model, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def predict(self, x_input):
        """
        Predict output for given input.

        :param x_input: Model input
        :return: Predictions of output layer
        """
        if self.base_model is None:
            predictions = self.top_model.predict(x_input)
        else:
            bottleneck_features = self.base_model.predict(x_input)
            predictions = self.top_model.predict(bottleneck_features)
        return predictions

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
