# import stuff
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import keras
from keras import layers
from tensorflow.keras.utils import plot_model

# load in the training and testing data
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# set image dimensions and batch size
img_dim = (48, 48)
batch_size = 32

# write an image generator
train_datagen = ImageDataGenerator(
    rescale=1./255, # rescale pixel values for greater stability
    horizontal_flip = True, # laterally invert images
    zoom_range = 0.1
)

test_datagen = ImageDataGenerator(rescale=1./255) # only need to rescale pixel values

# load images in from folder
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_dim,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_dim,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# second attempt using ResNet
# create residual block
def residual_block(x, filters, kernel_size=(3, 3), stride=1, activation='relu', kernel_initializer='he_normal', regularization=None):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=regularization)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=regularization)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
      shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same',
                                  kernel_initializer=kernel_initializer, kernel_regularizer=regularization)(shortcut)
      shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation(activation)(x)
    return x

# define model here

def create_resnet_model(input_shape, output_shape, regularization=None):
    x = keras.layers.Input(shape=input_shape)
    # Initial Convolution and MaxPooling layers
    y = keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', strides=(2, 2),
                            kernel_initializer='he_normal', kernel_regularizer=regularization)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(y)
    # First set of residual blocks
    y = residual_block(y, 32, regularization=regularization)
    y = residual_block(y, 32, regularization=regularization)
    # Second set of residual blocks with increased filters
    y = residual_block(y, 64, stride=2, regularization=regularization)  # Downsample
    y = residual_block(y, 64, regularization=regularization)
    # Third set of residual blocks with further increased filters
    y = residual_block(y, 128, stride=2, regularization=regularization)  # Downsample
    y = residual_block(y, 128, regularization=regularization)
    # Fourth set of residual blocks with further increased filters
    y = residual_block(y, 192, stride=2, regularization=regularization)  # Downsample
    y = residual_block(y, 192, regularization=regularization)
    # Fifth set of residual blocks
    y = residual_block(y, 256, stride=2, regularization=regularization)  # Downsample
    y = residual_block(y, 256, regularization=regularization)
    # Global Average Pooling and Dropout
    y = keras.layers.GlobalAveragePooling2D()(y)
    y = keras.layers.Dropout(0.5)(y)
    # Final Dense layer with softmax activation
    y = keras.layers.Dense(output_shape, activation='softmax', kernel_initializer='he_normal')(y)
    model = keras.models.Model(inputs=x, outputs=y)
    # Compile model with an adaptive optimizer
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=5000,
        decay_rate=0.9
    )
    optimizer = keras.optimizers.Adam(learning_rate = lr_schedule)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# instantiate model
model = create_resnet_model(input_shape=(48, 48, 1), output_shape=7)

model.summary()

