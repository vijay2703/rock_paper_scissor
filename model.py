# NO NOT MODIFY THE CONTENTS OF THIS FILE UNLESS REQUIRED

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

CLASS_MAP = {
    "empty": 0,
    "rock": 1,
    "paper": 2,
    "scissors": 3
}

def mapper(value):
    return CLASS_MAP[value]

def create_model(input_shape, num_classes):
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The third convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The fifth convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def generate_model(X, y):

    dataset, labels = X, y

    # label encode the classes
    labels = list(map(mapper, labels))
    input_shape = (225, 225, 3)
    model = create_model(input_shape, len(CLASS_MAP))

    with tf.device('/device:CPU:0'):

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()

        # start training
        model.fit(np.array(dataset), np.array(labels), epochs=15)

        # save the model for later use
        model.save("rock-paper-scissors-model.h5")
