import os
import shutil
import subprocess

import librosa as lp

import numpy as np

import keras
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from preprocess_util import *

def run_preprocess(root, length, split, n_mfcc, n_mfcc_width, transfer = False):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            mfcc_data = []
            npy_file = directory + '_' + 'mfcc' + '_' + str(128) + '.npy'
            if os.path.isfile(os.path.join(subdir, directory, npy_file)):
                continue

            pr_file = "preprocess_transfer" if transfer else "preprocess"
            if not os.path.isdir(os.path.join(subdir, directory, "split", split)):
                subprocess.call([os.path.join('..', 'data_pro', pr_file),\
                                 os.path.join(subdir, directory), length, split])

            file_path = os.path.join(subdir, directory, "split", split, "wav")
            for filename in os.listdir(file_path):
                y, sr = lp.load(os.path.join(file_path, filename))
                mfcc = lp.feature.mfcc(y = y, sr = 16000, n_mfcc = 128)
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, n_mfcc_width)), mode='constant')
                if mfcc.shape != (n_mfcc, n_mfcc_width):
                    mfcc = mfcc[:, :n_mfcc_width]
                mfcc_data.append(mfcc)

            np.save(os.path.join(subdir, directory, npy_file), np.asarray(mfcc_data))
            # shutil.rmtree(os.path.join(subdir, directory, "split"), ignore_errors = True)
        break

def load_features(root, length, split, n_mfcc, n_mfcc_width, dist = False):
    mfcc_data = np.zeros((0, n_mfcc, n_mfcc_width))
    mfcc_label = []
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            if dist == True and len(directory.split(".")) != 2:
                continue
            npy_file = directory + '_' + 'mfcc' + '_' + str(128) + '.npy'
            mfcc = np.load(os.path.join(subdir, directory, npy_file))
            if mfcc.size == 0:
                continue
            mfcc_data = np.concatenate((mfcc_data, mfcc[:, :n_mfcc, :]))
            mfcc_label += mfcc.shape[0] * [directory.split('.')[0]]
        break
    return mfcc_data, mfcc_label

# Build the CNN for MFCC
def build_mfcc_model(input_shape, n_dense, output_labels):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu',
              input_shape = input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(n_dense, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_labels))
    model.add(Activation('softmax'))

    # Initialize RMSprop
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model
