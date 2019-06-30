import sys
import os
import random
import math
import tensorflow as tf
import numpy as np
np.random.seed(1)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image

class AI(object):
    def __init__(self):
        super(AI, self).__init__()

        self.init_default()

    def init_default(self):
        # from keras.datasets import mnist
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # x_train = x_train.reshape(-1, 1, 28, 28) / 255.
        # x_test = x_test.reshape(-1, 1, 28, 28) / 255.
        # y_train = np_utils.to_categorical(y_train, num_classes=10)
        # y_test = np_utils.to_categorical(y_test, num_classes=10)

        self.model = Sequential()

        self.model.add(Convolution2D(
            batch_input_shape=(None, 1, 28, 28),
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',  # Padding method
            data_format='channels_first',
        ))
        self.model.add(Activation('relu'))

        # Pooling layer 1 (max pooling) output shape (32, 14, 14)
        self.model.add(MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
            data_format='channels_first',
        ))

        # Conv layer 2 output shape (64, 14, 14)
        self.model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
        self.model.add(Activation('relu'))

        # Pooling layer 2 (max pooling) output shape (64, 7, 7)
        self.model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

        # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))

        # Fully connected layer 2 to shape (10) for 10 classes
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

        del self.model  # deletes the existing model
        self.model = load_model('./model/number_model.h5')
        self.graph = tf.get_default_graph()
        print('Reloaded')

        # model.fit(x_train, y_train, epochs=2, batch_size=64)
        # model.save('./model/model.h5')  # HDF5 file, you have to pip3 install h5py if don't have it
        # print('Saved')

    def predict_image_with_path(self, file_path):
        try:
            im = Image.open(file_path).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)  # resize the image
            im = np.array(im)  # convert to an array
            im = np.ones(im.shape)*255 - im
            im2 = im / np.max(im).astype(float)  # normalise input

            # plt.imshow(im2, cmap='gray')
            # plt.title('My Image')
            # plt.axis('off')
            # plt.show()

            with self.graph.as_default():
                test_image = np.reshape(im2, [1, 1, 28, 28])  # reshape it to our input placeholder shape
                p_ = self.model.predict(test_image).argmax(axis=1)
                return '我覺得是{}!'.format(p_[0])
            return '哎呀呀失誤了...'
        except:
            return '哎呀呀我失誤了...'