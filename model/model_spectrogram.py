import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from preprocess_util import *

# Preprocess script is run with given arguments.
# Saves the spectrogram files on the disk.
# Note: Change the filename below to 'preprocess_distributed' to
# run the preprocess step on distributed samples across multiple directories.
def run_preprocess_spect(root, length, split, transfer = False):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            pr_file = "preprocess_transfer" if transfer else "preprocess"
            if not os.path.isdir(os.path.join(subdir, directory, "split", split)):
                subprocess.call([os.path.join('..', 'data_pro', pr_file),\
                                 os.path.join(subdir, directory), length, split, "s"])
        break

# Loads the saved spectrogram files into the main memory.
def load_features_spect(root, split):
    spect_data = []
    spect_label = []
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            file_path = os.path.join(subdir, directory, "split", split, "spect")
            for filename in os.listdir(file_path):
                x = plt.imread(os.path.join(file_path, filename))
                spect_data.append(x)
                spect_label.append(directory.split('.')[0])
        break
    return spect_data, spect_label

# Use this when you intend to load only certain number of
# spectrogram files for each sample.
def load_partial_features(root, split, start, count):
    spect_data = []
    spect_label = []
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            s_l = start
            while not os.path.isfile(os.path.join(subdir, directory, "split", split, "spect", '%03d' % s_l + '.wav.png')):
                s_l -= count
            for i in range(s_l, s_l + count):
                file_name = os.path.join(subdir, directory, "split", split, "spect", '%03d' % i + '.wav.png')
                if os.path.isfile(file_name):
                    x = plt.imread(file_name)
                    spect_data.append(x)
                    spect_label.append(directory)
        break
    return spect_data, spect_label

# Build the CNN for spectrogram
def build_spectrogram_model(input_shape, output_labels):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same',
                     input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(output_labels))
    model.add(Activation('softmax'))

    # Initialize RMSprop
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
