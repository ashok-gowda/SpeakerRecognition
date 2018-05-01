import os

from sklearn.model_selection import train_test_split

from model_mfcc import *
from utils import *

n_mfcc = 128
n_mfcc_width = 432
window_size = 10
audio_len = 90
data_dir = os.path.join('..', 'audio-train-new')
mfcc_shape = (n_mfcc, n_mfcc_width, 1)
n_samples = 112

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
    model.save_weights(os.path.join('..', 'neural-net-weights', \
                                    'mfcc_model_weights_' + str(n_mfcc) + '_' + \
                                        str(audio_len) + '_' + str(window_size) + '.h5'))
    
    print("Successfully completed.")

if __name__ == "__main__":
    main()

