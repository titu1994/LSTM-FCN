from keras.models import Model
from keras.layers import Input, PReLU, Dense,Dropout, LSTM, Bidirectional, multiply, concatenate
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute
from keras import backend as K
from phased_lstm_keras.PhasedLSTM import PhasedLSTM

from utils.constants import MAX_NB_WORDS_LIST, MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualise_attention, hyperparameter_search_over_model

DATASET_INDEX = 12
OUTPUT_DIM = 1000

MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
MAX_NB_WORDS = MAX_NB_WORDS_LIST[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

ATTENTION_CONCAT_AXIS = 1 # 1 = temporal, -1 = spatial
TRAINABLE = True

def generate_model():

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = attention_block(ip, id=1)
    x = concatenate([ip, x], axis=ATTENTION_CONCAT_AXIS)

    x = LSTM(64)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    cnn_count = 0
    for layer in model.layers:
        if layer.__class__.__name__ in ['Conv1D',
                                        'BatchNormalization',
                                        'PReLU']:
            if layer.__class__.__name__ == 'Conv1D':
                cnn_count += 1

            if cnn_count == 3:
                break
            else:
                set_trainable(layer, TRAINABLE)

    model.summary()

    #model.load_weights('./weights/ecg200_weights.h5')

    return model


def attention_block(inputs, id):
    # input shape: (batch_size, time_step, input_dim)
    # input shape: (batch_size, max_sequence_length, lstm_output_dim)
    input_dim = K.int_shape(inputs)[-1]
    x = Dense(input_dim, activation='softmax', name='attention_dense_%d' % id)(inputs)
    x = multiply([inputs, x])
    return x



if __name__ == "__main__":
    model = generate_model()

    train_model(model, DATASET_INDEX, dataset_prefix='ecg200', epochs=4000, batch_size=128)

    evaluate_model(model, DATASET_INDEX, dataset_prefix='ecg200', batch_size=128)
