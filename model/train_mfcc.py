import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

from model_mfcc import *
from utils import *

n_mfcc = 128
n_mfcc_width = 432
window_size = 10
audio_len = 90
data_dir = os.path.join('..', 'audio-train-new')
mfcc_shape = (n_mfcc, n_mfcc_width, 1)
n_samples = 112

def plot_model(model):
    plot_model(model, to_file='mfcc_model.png', show_shapes=True)
    
def plot_val_acc(train_result):
    plt.style.use('dark_background')
    plt.plot(train_result.history['acc'], color="#5599FF")
    plt.plot(train_result.history['val_acc'], color="#55FF99")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    # Generate MFCC features and save them on the disk.
    print("Running preprocess..")
    run_preprocess(data_dir, str(audio_len), str(window_size), n_mfcc, n_mfcc_width)
    
    # Load saved MFCC coefficients from npy files
    print("Loading MFCC data..")
    X, y = load_features(data_dir, str(audio_len), str(window_size), n_mfcc, n_mfcc_width)
        
    # Reshape and one hot encode the data
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y_norm = one_hot_encode(y)
    
    # Split the samples and training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_norm, test_size=0.3, random_state=64)
    
    # Build the CNN
    print("Building the model..")
    model = build_mfcc_model(mfcc_shape, n_mfcc / 2, n_samples)
    
    # Train the model.
    model.fit(np.array(X_train), y_train,
          batch_size=16,
          epochs=30,
          verbose=1,
          shuffle = True,
         validation_data=(np.array(X_test), y_test))
    
    # Save the trained model weights
    model.save_weights('mfcc_model_weights_' + str(audio_len) + '_' + \
                       str(window_size) + '-' + str(n_mfcc) + '.h5')
    
    print("Successfully completed.")

if __name__ == "__main__":
    main()

