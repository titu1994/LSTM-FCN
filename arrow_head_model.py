from keras.models import Model
from keras.layers import Input, PReLU, Dense,Dropout, LSTM, Bidirectional, multiply, concatenate
from phased_lstm_keras.PhasedLSTM import PhasedLSTM

from utils.constants import MAX_NB_WORDS_LIST, MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualise_attention

DATASET_INDEX = 1
TRAINABLE = True

MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
MAX_NB_WORDS = MAX_NB_WORDS_LIST[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

def generate_model():

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = attention_block(ip)
    x = concatenate([ip, x], axis=-1)

    x = Bidirectional(LSTM(512, trainable=TRAINABLE))(x)
    #x = PhasedLSTM(512)(x)

    x = Dense(1024, activation='linear')(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)

    x = Dense(1024, activation='linear')(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    for layer in model.layers[:-4]:
        set_trainable(layer, TRAINABLE)

    model.summary()

    #model.load_weights('weights/arrow_head_weights.h5')

    return model


def attention_block(inputs):
    # input shape: (batch_size, time_step, input_dim)
    # input shape: (batch_size, max_sequence_length, lstm_output_dim)
    x = Dense(MAX_SEQUENCE_LENGTH, activation='softmax', name='attention_dense')(inputs)
    x = multiply([inputs, x])
    return x


if __name__ == "__main__":
    model = generate_model()

    train_model(model, DATASET_INDEX, dataset_prefix='arrow_head', epochs=100, batch_size=128,
                val_subset=175)

    evaluate_model(model, DATASET_INDEX, dataset_prefix='arrow_head', batch_size=128,
                  test_data_subset=175)

    #visualise_attention(model, DATASET_INDEX, dataset_prefix='arrow_head', layer_name='attention_dense')


