from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# Convert the vec to one-hot encoded vectors.
def one_hot_encode(vec):
    l_enc = LabelEncoder()

    l_enc.fit(vec)
    vec_enc = l_enc.transform(vec)
    vec_norm = np_utils.to_categorical(vec_enc)
    
    return vec_norm
