import os
import shutil
import subprocess

import numpy as np

import keras
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model, Sequential

def load_features_hybrid(root, length, split, n_coeff, n_mfcc, n_mfcc_width, dist = False):
    lpcc_data=[]
    mfcc_data = np.zeros((0, n_mfcc, n_mfcc_width))
    label = []
    for subdir, dirs, files in os.walk(root):
        count = 0
        for directory in dirs:
            if dist == True and len(directory.split(".")) != 2:
                continue

            npy_file_lp = directory + '_lpcc_' + str(n_coeff) + '_' + length + '_' + split + '.npy'
            npy_file_mf = directory + '_mfcc_' + str(128) + '_' + length + '_' + split + '.npy'

            if(count==0):
                lpcc_data= np.load(os.path.join(subdir, directory, npy_file_lp))
            else:
                lpcc=np.load(os.path.join(subdir, directory, npy_file_lp))
                lpcc_data=np.vstack((lpcc_data,lpcc))
            count+=1

            mfcc = np.load(os.path.join(subdir, directory, npy_file_mf))
            if mfcc.size == 0:
                continue
            mfcc_data = np.concatenate((mfcc_data, mfcc[:, :n_mfcc, :]))
            label += mfcc.shape[0] * [directory.split('.')[0]]
        break
    return lpcc_data, mfcc_data, label

def build_model_hybrid(input_shape_lp, input_shape_mf, n_dense_lp, n_dense_mf, output_labels):
    model_lp = Sequential()
    model_lp.add(Conv2D(32, kernel_size=(2, 2), activation='relu',
                     input_shape=input_shape_lp))
    model_lp.add(MaxPooling2D(pool_size=(2, 2)))
    model_lp.add(Activation('relu'))
    model_lp.add(Dropout(0.25))

    model_lp.add(Conv2D(32, kernel_size=(2, 2), padding='same'))
    model_lp.add(Activation('relu'))
    model_lp.add(MaxPooling2D(pool_size=(2, 2)))
    model_lp.add(Dense(n_dense_lp))
    model_lp.add(Dropout(0.25))
    model_lp.add(Flatten())

    input_lp = Input(shape=input_shape_lp)
    layer_lp = model_lp(input_lp)

    model_mf = Sequential()
    model_mf.add(Conv2D(32, kernel_size=(2, 2), activation='relu',
                     input_shape=input_shape_mf))
    model_mf.add(MaxPooling2D(pool_size=(2, 2)))
    model_mf.add(Activation('relu'))
    model_mf.add(Dropout(0.25))

    model_mf.add(Conv2D(32, kernel_size=(2, 2), padding='same'))
    model_mf.add(Activation('relu'))
    model_mf.add(MaxPooling2D(pool_size=(2, 2)))
    model_mf.add(Dense(n_dense_mf, activation='relu'))
    model_mf.add(Dropout(0.25))
    model_mf.add(Flatten())

    input_mf = Input(shape=input_shape_mf)
    layer_mf = model_mf(input_mf)

    merged = keras.layers.concatenate([layer_lp, layer_mf])
    output = Dense(output_labels, activation='softmax')(merged)

    model = Model(inputs=[input_lp, input_mf], outputs=output)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

