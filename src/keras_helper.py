import numpy as np
import os
import h5py

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend
from keras.engine import topology
from keras.applications.vgg16 import VGG16

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


class AmazonKerasClassifier:
    def __init__(self):
        self.losses = []
        self.classifier = Sequential()

    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))

        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(128, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(256, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))


    def add_flatten_layer(self):
        self.classifier.add(Flatten())


    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def train_model(self, x_train, y_train, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size)

        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(X_valid, y_valid),
                            callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
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

class VGG16DenseRetrainer:
    def __init__(self, path_bottleneck_feat_trn='bottleneck_features_train.npy',
                 path_bottleneck_feat_val='bottleneck_features_validation.npy'):
        self.losses = []
        self.bottleneck_model = Sequential()
        self.top_model = Sequential()
        self.path_bottleneck_feat_trn = path_bottleneck_feat_trn
        self.path_bottleneck_feat_val = path_bottleneck_feat_val
        self.bottleneck_feat_trn = None
        self.bottleneck_feat_val = None

    def build_vgg16(self, img_size, img_channels, n_classes):
        img_width = img_size[0]
        img_height = img_size[1]

        self.bottleneck_model = VGG16(include_top=False, weights='imagenet',
              input_tensor=None, input_shape=(img_width, img_height, img_channels),
              pooling=None,
              classes=n_classes)

    def build_vgg16_old(self, img_size=(32, 32), img_channels=3):
        img_width = img_size[0]
        img_height = img_size[1]

        #self.bottleneck_model.add(BatchNormalization(input_shape=(*img_size, img_channels)))
        """
        self.bottleneck_model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, img_channels)))

        self.bottleneck_model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
        self.bottleneck_model.add(MaxPooling2D(pool_size=2))

        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
        self.bottleneck_model.add(MaxPooling2D(pool_size=2))

        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
        self.bottleneck_model.add(MaxPooling2D(pool_size=2))

        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
        self.bottleneck_model.add(MaxPooling2D(pool_size=2))

        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
        self.bottleneck_model.add(ZeroPadding2D((1, 1)))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
        self.bottleneck_model.add(MaxPooling2D(pool_size=2))
        """

        # Block 1
        self.bottleneck_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',input_shape=(img_width, img_height, img_channels)))
        self.bottleneck_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
        self.bottleneck_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        self.bottleneck_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
        self.bottleneck_model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
        self.bottleneck_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        self.bottleneck_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
        self.bottleneck_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
        self.bottleneck_model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
        self.bottleneck_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
        self.bottleneck_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
        self.bottleneck_model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
        self.bottleneck_model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    def load_vgg16_weights(self, filepath, by_name=False):
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # Legacy support
        layers = self.bottleneck_model.layers
        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

    def load_vgg16_weights_old(self, weights_path):
        """Load vgg16 weights into bottleneck model."""
        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(self.bottleneck_model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            self.bottleneck_model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

    def predict_bottleneck_features(self, x_train, y_train, validation_split_size=0.2):
        """Runs input through vgg16 architecture to generate and save bottleneck features."""
        X_train, X_valid, _, _ = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size)

        self.bottleneck_feat_trn = self.bottleneck_model.predict(X_train)
        np.save(open(self.path_bottleneck_feat_trn, 'wb'), self.bottleneck_feat_trn)

        self.bottleneck_feat_val = self.bottleneck_model.predict(X_valid)
        np.save(open(self.path_bottleneck_feat_val, 'wb'), self.bottleneck_feat_val)

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def build_top_model(self, n_classes):
        self.top_model.add(Flatten(input_shape=self.bottleneck_feat_trn.shape[1:]))
        self.top_model.add(Dense(256, activation='relu'))
        self.top_model.add(Dropout(0.5))
        self.top_model.add(Dense(n_classes, activation='sigmoid'))

    def train_top_model(self, x_train, y_train, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2,
                    train_callbacks=()):
        """Builds and trains top model."""
        history = LossHistory()

        _, _, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size)

        opt = Adam(lr=learn_rate)

        self.top_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.top_model.fit(self.bottleneck_feat_trn, y_train,
                           epochs=epoch,
                           batch_size=batch_size,
                           verbose=1,
                           validation_data=(self.bottleneck_feat_val, y_valid),
                           callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.top_model, self.bottleneck_feat_val, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_top_weights(self, weight_file_path):
        self.top_model.save_weights(weight_file_path)

    def load_top_weights(self, weight_file_path):
        self.top_model.load_weights(weight_file_path)

    def predict(self, x_test):
        bottleneck_features = self.bottleneck_model.predict(x_test)
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
