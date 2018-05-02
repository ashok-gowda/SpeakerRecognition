import os

from sklearn.model_selection import train_test_split

from model_lpcc import *
from utils import *

n_lpcc = 49 # '(n_lpcc + 1)' must be divisible by 5.
window_size = 10
audio_len = 600
data_dir = os.path.join('..', 'audio-train-new')
lpcc_shape = (10, (n_lpcc + 1) / 5, 1)
n_samples = 112

def main():
    # Generate LPCC features and save them on the disk.
    print("Running preprocess..")
    run_preprocess(data_dir, str(audio_len), str(window_size), n_lpcc)
    
    # Load saved LPCC coefficients from npy files
    print("Loading LPCC data..")
    X, y = load_features(data_dir, str(audio_len), str(window_size), n_lpcc)
        
    # Reshape and one hot encode the data
    X = X.reshape(X.shape[0], 10, -1, 1)
    y_norm = one_hot_encode(y)
    
    # Split the samples and training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_norm, test_size=0.3, random_state=64)
    
    # Build the CNN
    print("Building the model..")
    model = build_lpcc_model(lpcc_shape, n_lpcc + 1, n_samples)
    
    # Train the model.
    train_result = model.fit(np.array(X_train), y_train,
          batch_size=16,
          epochs=600,
          verbose=1,
          shuffle = True,
         validation_data=(np.array(X_test), y_test))
    
    # Save the trained model weights
    model.save_weights(os.path.join('..', 'neural-net-weights', \
                                    'lpcc_model_weights_' + str(n_lpcc) + '_' + \
                                        str(audio_len) + '_' + str(window_size) + '.h5'))
    
    print("Successfully completed.")

if __name__ == "__main__":
    main()

