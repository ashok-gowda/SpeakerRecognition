# ### Spectrogram based Transfer Learning
import os

import numpy as np

from sklearn.model_selection import train_test_split

from model_spectrogram import *
from utils import *

window_size = 12
audio_len = 84
data_dir = os.path.join('..', 'audio-train-new')
n_samples = 120
spect_shape = (540, 960, 3)
c_batch_size = 32

# Train model for 5 epochs.
# Note: As per our observations, spectrogram reaches good accuracy within 5 epochs
def train_model(model, X_train, y_train, X_test, y_test, epochs, batch):
    for epoch in range(epochs):
        model.fit(X_train, y_train,
                  batch_size = batch,
                  epochs = 1,
                  verbose = 1,
                 validation_data = (X_test, y_test))
        # Save the weights
        model.save_weights('spect_model_weights_' + str(epoch) + '.h5')

def main():
    # Generate spectrograms and save them on the disk.
    print("Running preprocess")
    run_preprocess_spect(data_dir, str(audio_len), str(window_size))

    # Load saved spectrograms
    print("Loading spectrogram files")
    X, y = load_features_spect(data_dir, str(window_size))

    cleanup_split(data_dir)

    # Get the one-hot encoded vectors
    y_norm = one_hot_encode(y)

    # Split the samples and training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_norm, test_size = 0.3, random_state = 42)

    # Delete the unused variables, to save memory
    del X, y, y_norm

    # Build the CNN
    print("Building the model..")
    model = build_spectrogram_model(spect_shape, n_samples)

    # Train the model using the spectrogram images.
    print("Training the model..")
    train_model(model, np.array(X_train), y_train, np.array(X_test), y_test, 5, c_batch_size)

    print("Successfully completed.")

if __name__ == "__main__":
    main()

