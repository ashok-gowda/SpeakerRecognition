import os

from sklearn.model_selection import train_test_split

from model_hybrid import *
from model_lpcc import *
from model_mfcc import *
from utils import *

n_lpcc = 49 # '(n_lpcc + 1)' must be divisible by 5.
n_mfcc = 64
n_mfcc_width = 430
window_size = 10
audio_len = 500
data_dir = os.path.join('..', 'audio-train-new')
lpcc_shape = (10, (n_lpcc + 1) / 5, 1)
mfcc_shape = (n_mfcc, n_mfcc_width, 1)
n_samples = 112

# Split the combined data into train and test sets.
# Then reorganize them to LPCC and MFCC data
def split_and_separate(X, y):
    # Split the samples and training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64)

    z_train = np.hsplit(X_train, [X_lp.shape[1], X_mf.shape[0] * X_mf.shape[1]])
    X_lp_train = z_train[0].reshape(z_train[0].shape[0], 10, -1, 1)
    X_mf_train = z_train[1].reshape(z_train[1].shape[0], X_mf.shape[1], -1, 1)

    z_test = np.hsplit(X_test, [X_lp.shape[1], X_mf.shape[0] * X_mf.shape[1]])
    X_lp_test = z_test[0].reshape(z_test[0].shape[0], 10, -1, 1)
    X_mf_test = z_test[1].reshape(z_test[1].shape[0], X_mf.shape[1], -1, 1)

    return X_lp_train, X_lp_test, X_mf_train, X_mf_test, y_train, y_test

def main():
    # Generate LPCC features and save them on the disk.
    print("Running preprocess..")
    run_preprocess_lpcc(data_dir, str(audio_len), str(window_size), n_lpcc)
    run_preprocess_mfcc(data_dir, str(audio_len), str(window_size), n_mfcc, n_mfcc_width)

    # Load saved LPCC & MFCC coefficients from npy files
    print("Loading LPCC & MFCC data..")
    X_lp, X_mf, y = load_features_hybrid(data_dir, str(audio_len), str(window_size), \
                                         n_lpcc, n_mfcc, n_mfcc_width)

    # Reshape to stack LPCC & MFCC arrays together and then one hot encode the labels
    X = np.hstack((X_lp, X_mf.reshape(X_mf.shape[0], -1)))
    y_norm = one_hot_encode(y)

    # Split into train/test data and separate the LPCC & MFCC vectors
    X_lp_train, X_lp_test, X_mf_train, X_mf_test, y_train, y_test = split_and_separate(X, y_norm)

    # Build the CNN
    print("Building the model..")
    model = build_model_hybrid(lpcc_shape, mfcc_shape, n_lpcc + 1, n_mfcc / 2, n_samples)

    # Train the model.
    train_result = model.fit([X_lp_train, X_mf_train], y_train,
          batch_size=128,
          epochs=160,
          verbose=1,
          shuffle = True,
         validation_data=([X_lp_test, X_mf_test], y_test))

    # Save the trained model weights
    model.save_weights(os.path.join('..', 'neural-net-weights', \
                                    'hybrid_model_weights_' + str(n_lpcc) + '_' + str(n_mfcc) + '_' + \
                                        str(audio_len) + '_' + str(window_size) + '.h5'))

    print("Successfully completed.")

if __name__ == "__main__":
    main()

