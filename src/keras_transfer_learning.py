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
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import optimizers

# vgg16 weights require th input ordering
#from keras import backend as K
#K.set_image_dim_ordering('th')

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class TransferModel:
    """Class for retraining VGG16 architecture.

    Supports retraining dense top_model:
        classifier = VGG16DenseRetrainer(...)
        classifier.build_vgg16(...)
        classifier.predict_bottleneck_features(...)
        classifier.build_top_model(...)
        classifier.train_top_model(...)
    Supports retraining dense top_model with layers of VGG16 architecture. Note that this requires training vanilla
    dense top_model first:
        *dense top model steps*
        classifier.split_fine_tuning_models(...)
        classifier.predict_bottleneck_features(...)
        classifier.fine_tune_full_model(...)
    When training over multiple parameters in loop make sure to load original top_model weights stored by
    build_full_model(...) model call:
        *dense top model steps*
        init_top_weights = classifier.split_fine_tuning_models(...)
        classifier.predict_bottleneck_features(...)
        for param in params:
            classifier.set_top_model_weights(init_top_weights)
            classifier.fine_tune_full_model(...)

    """
    def __init__(self, path_bottleneck_feat_trn='bottleneck_features_train.npy',
                 path_bottleneck_feat_val='bottleneck_features_validation.npy'):
        self.losses = []
        self.base_model = None
        self.top_model = Sequential()
        self.full_model = None
        self.path_bottleneck_feat_trn = path_bottleneck_feat_trn
        self.path_bottleneck_feat_val = path_bottleneck_feat_val
        self.bottleneck_feat_trn = None
        self.bottleneck_feat_val = None
        self.classification_threshold = 0.2

    def build_vgg16(self, img_size, img_channels, n_classes):
        img_width = img_size[0]
        img_height = img_size[1]

        self.base_model = VGG16(include_top=False, weights='imagenet',
              input_tensor=None, input_shape=(img_width, img_height, img_channels),
              pooling=None,
              classes=n_classes)

    def load_vgg16_weights_old(self, weights_path):
        """Load vgg16 weights into bottleneck model."""
        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(self.base_model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            self.base_model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

    def predict_bottleneck_features(self, X_train, X_valid, validation_split_size=0.2):
        """Runs input through vgg16 architecture to generate and save bottleneck features."""
        self.bottleneck_feat_trn = self.base_model.predict(X_train)
        np.save(open(self.path_bottleneck_feat_trn, 'wb'), self.bottleneck_feat_trn)

        self.bottleneck_feat_val = self.base_model.predict(X_valid)
        np.save(open(self.path_bottleneck_feat_val, 'wb'), self.bottleneck_feat_val)

    def _get_fbeta_score(self, classifier, X_valid, y_valid, threshold=None):
        if threshold is None:
            threshold = self.classification_threshold
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > threshold, beta=2, average='samples')

    def fit_classification_threshold(self, classifier, x_input, y_true):
        prediction = classifier.predict(x_input)

        best_f2 = 0
        for i in range(1,6):
            threshold = float(i)/10
            f2 = fbeta_score(y_true, np.array(prediction) > threshold, beta=2, average='samples')
            if f2 > best_f2:
                best_f2 = f2
                self.classification_threshold = threshold

        return self.classification_threshold

    def get_fbeta_score_valid(self, y_valid):
        fbeta_score = self._get_fbeta_score(self.top_model, self.bottleneck_feat_val, y_valid)
        return fbeta_score

    def build_top_model(self, n_classes, n_dense=256, dropout_rate=0.3):
        self.top_model.add(Flatten(input_shape=self.base_model.output_shape[1:]))
        self.top_model.add(Dense(n_dense, activation='relu'))
        self.top_model.add(Dropout(dropout_rate))
        self.top_model.add(Dense(n_classes, activation='sigmoid'))

    def train_top_model(self, y_train, y_valid, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2,
                    train_callbacks=()):
        """Builds and trains top model."""
        history = LossHistory()

        opt = Adam(lr=learn_rate)

        self.top_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.top_model.fit(self.bottleneck_feat_trn, y_train,
                           epochs=epoch,
                           batch_size=batch_size,
                           verbose=2,
                           validation_data=(self.bottleneck_feat_val, y_valid),
                           callbacks=[history, *train_callbacks, earlyStopping])
        #determine classification threshold
        self.fit_classification_threshold(self.top_model, self.bottleneck_feat_trn, y_train)
        fbeta_score = self._get_fbeta_score(self.top_model, self.bottleneck_feat_val, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_top_weights(self, weight_file_path):
        self.top_model.save_weights(weight_file_path)

    def load_top_weights(self, weight_file_path):
        self.top_model.load_weights(weight_file_path)

    def set_top_weights(self, weights):
        self.top_model.set_weights(weights)

    def get_top_weights(self):
        return self.top_model.get_weights()

    #def load_full_weights(self, weight_file_path):
    #    self.full_model.load_weights(weight_file_path)

    def split_fine_tuning_models(self, split_layer_id):
        """Splits full model at split_layer_id and returns new top_model weights.

        Full model consists of previous base_model plus top_model. New base_model goes until split_layer_id. New
        top_model starts at split_layer_id. The top_model weights are stored and returned as starting point for future
        retraining.
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

    def fine_tune_full_model(self, y_train, y_valid, learn_rate=0.001, momentum=0.9, epoch=5, batch_size=128, validation_split_size=0.2,
                             train_callbacks=()):
        """Retrain top model and last vgg16 conv block with light weight updates."""

        history = LossHistory()

        self.top_model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=learn_rate, momentum=momentum),
                      metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        self.top_model.fit(self.bottleneck_feat_trn, y_train,
                           epochs=epoch,
                           batch_size=batch_size,
                           verbose=2,
                           validation_data=(self.bottleneck_feat_val, y_valid),
                           callbacks=[history, *train_callbacks, earlyStopping])
        self.fit_classification_threshold(self.top_model, self.bottleneck_feat_trn, y_train)
        fbeta_score = self._get_fbeta_score(self.top_model, self.bottleneck_feat_val, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def predict(self, x_test):
        bottleneck_features = self.base_model.predict(x_test)
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
        backend.clear_session()
