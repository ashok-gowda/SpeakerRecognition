import os
import shutil
import subprocess

import librosa as lp
from scikits.talkbox import lpc

import numpy as np

import keras
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from preprocess_util import *

def convert_to_lpc(filename, n_coeff):
    wave, sr = lp.load(filename, mono = True, sr = 16000)
    lpc_signal=lpc(wave, n_coeff)
    return np.hstack((lpc_signal[0], lpc_signal[1], lpc_signal[2]))

def run_preprocess(root, length, split, n_coeff, transfer = False):
    for subdir, dirs, files in os.walk(root):
        for directory in dirs:
            lpcc_data = []
            npy_file = directory + '_lpcc_' + str(n_coeff) + '_' + length + '_' + split + '.npy'
            if os.path.isfile(os.path.join(subdir, directory, npy_file)):
                continue

            pr_file = "preprocess_transfer" if transfer else "preprocess"
            if not os.path.isdir(os.path.join(subdir, directory, "split", split)):
                subprocess.call([os.path.join('..', 'data_pro', pr_file),\
                                 os.path.join(subdir, directory), length, split])

            file_path = os.path.join(subdir, directory, "split", split, "wav")
            for filename in os.listdir(file_path):
                lpcc_data.append(convert_to_lpc(os.path.join(file_path, filename), n_coeff))

            if np.asarray(lpcc_data).shape[0] == 0:
                continue

            np.save(os.path.join(subdir, directory, npy_file), np.asarray(lpcc_data))
        break

def load_features(root, length, split, n_coeff):
    lpcc_data=[]
    lpcc_label = []
    for subdir, dirs, files in os.walk(root):
        count=0
        for directory in dirs:
            npy_file = directory + '_lpcc_' + str(n_coeff) + '_' + length + '_' + split + '.npy'

            if(count==0):
                lpcc_data= np.load(os.path.join(subdir, directory, npy_file))
                lpcc_label=lpcc_data.shape[0]*[directory.split('.')[0]]
            else:
                lpcc=np.load(os.path.join(subdir, directory, npy_file))
                lpcc_data=np.vstack((lpcc_data,lpcc))
                lpcc_label += lpcc.shape[0] * [directory.split('.')[0]]
            count += 1
        break
    return lpcc_data, lpcc_label

# Build the CNN for LPCC 
def build_lpcc_model(input_shape, n_dense, output_labels):
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
