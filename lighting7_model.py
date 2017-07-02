from keras.models import Model
from keras.layers import Input, PReLU, Dense, Dropout, LSTM, Embedding, BatchNormalization, Bidirectional, multiply
from phased_lstm_keras.PhasedLSTM import PhasedLSTM

from utils.constants import MAX_NB_WORDS_LIST, MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, MaskablePermute

DATASET_INDEX = 4
OUTPUT_DIM = 1000
TRAINABLE = True

MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
MAX_NB_WORDS = MAX_NB_WORDS_LIST[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]


def generate_model():
    ip = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding = Embedding(input_dim=MAX_NB_WORDS, output_dim=OUTPUT_DIM,
                          mask_zero=True, input_length=MAX_SEQUENCE_LENGTH)(ip)

    embedding = attention_block(embedding)

    #x = Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, trainable=TRAINABLE))(embedding)
    x = PhasedLSTM(512, dropout=0.2, recurrent_dropout=0.2, implementation=2)(embedding)

    x = BatchNormalization()(x)

    x = Dense(1024, activation='linear')(x)
    x = PReLU()(x)

    x = BatchNormalization()(x)

    x = Dense(1024, activation='linear')(x)
    x = PReLU()(x)

    x = Dropout(0.2)(x)

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    for layer in model.layers[:-4]:
        set_trainable(layer, TRAINABLE)

    model.summary()

    #model.load_weights('weights/lighting7_weights - 5068 attn.h5')

    return model

def attention_block(inputs):
    # input shape: (batch_size, time_step, input_dim)
    # input shape: (batch_size, max_sequence_length, lstm_output_dim)
    x = MaskablePermute((2, 1))(inputs) # (batch_size, lstm_output_dim, max_sequence_length)
    x = Dense(MAX_SEQUENCE_LENGTH, activation='softmax', name='attention_dense')(x)
    x = MaskablePermute((2, 1), name='attention_vector')(x) # (batch_size, max_sequence_length, lstm_output_dim)
    x = multiply([inputs, x])
    return x


if __name__ == "__main__":
    model = generate_model()

    train_model(model, DATASET_INDEX, dataset_prefix='lighting7', epochs=100, batch_size=128,
                val_subset=73)

    evaluate_model(model, DATASET_INDEX, dataset_prefix='lighting7', batch_size=128,
                  test_data_subset=73)


