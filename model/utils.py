from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.utils import plot_model
import matplotlib.pyplot as plt

# Transform text labels into integer labels
def encode(vec):
    l_enc = LabelEncoder()

    l_enc.fit(vec)
    vec_enc = l_enc.transform(vec)

    return vec_enc

# Convert the vec to one-hot encoded vectors.
def one_hot_encode(vec):
    vec_enc = encode(vec)
    vec_norm = np_utils.to_categorical(vec_enc)

    return vec_norm

# Plot the model architecture and save it to a file
def plot_model(model):
    plot_model(model, to_file='model.png', show_shapes=True)

# Plot the train/test accuracy against no. of epochs
def plot_val_acc(train_result):
    plt.style.use('dark_background')
    plt.plot(train_result.history['acc'], color="#5599FF")
    plt.plot(train_result.history['val_acc'], color="#55FF99")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

