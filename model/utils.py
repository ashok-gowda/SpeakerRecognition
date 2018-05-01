from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

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
