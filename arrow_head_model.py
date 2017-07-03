from keras.models import Model
from keras.layers import Input, PReLU, Dense,Dropout, LSTM, Bidirectional, multiply, concatenate
from keras import backend as K
from phased_lstm_keras.PhasedLSTM import PhasedLSTM

from utils.constants import MAX_NB_WORDS_LIST, MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualise_attention, hyperparameter_search_over_model

DATASET_INDEX = 1
TRAINABLE = True

MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
MAX_NB_WORDS = MAX_NB_WORDS_LIST[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

def generate_model():

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    a = attention_block(ip, id=1)
    x = concatenate([ip, a], axis=-1)

    x = Bidirectional(LSTM(128, trainable=TRAINABLE))(x)
    #x = PhasedLSTM(512)(x)

    # x = Dense(1024, activation='linear')(x)
    # x = PReLU()(x)
    # x = Dropout(0.2)(x)

    # x = Dense(1024, activation='linear')(x)
    # x = PReLU()(x)
    x = Dropout(0.2)(x)

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    for layer in model.layers[:-4]:
        set_trainable(layer, TRAINABLE)

    model.summary()

    return model


def attention_block(inputs, id):
    # input shape: (batch_size, time_step, input_dim)
    # input shape: (batch_size, max_sequence_length, lstm_output_dim)
    input_dim = K.int_shape(inputs)[-1]
    x = Dense(input_dim, activation='softmax', name='attention_dense_%d' % id)(inputs)
    x = multiply([inputs, x])
    return x


def gridsearch_model_gen(lstm_cells=512, dense_neurons=1024, dropout=0.2):
    K.clear_session()

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = attention_block(ip, 1)
    x = concatenate([ip, x], axis=-1)

    x = Bidirectional(LSTM(lstm_cells, trainable=TRAINABLE))(x)

    x = Dense(dense_neurons, activation='linear')(x)
    x = PReLU()(x)
    x = Dropout(dropout)(x)

    x = Dense(dense_neurons, activation='linear')(x)
    x = PReLU()(x)
    x = Dropout(dropout)(x)

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = generate_model()

    train_model(model, DATASET_INDEX, dataset_prefix='arrow_head', epochs=150, batch_size=128,
                val_subset=175)

    evaluate_model(model, DATASET_INDEX, dataset_prefix='arrow_head', batch_size=128,
                  test_data_subset=175)

    #visualise_attention(model, DATASET_INDEX, dataset_prefix='arrow_head', layer_name='attention_dense')

    # hyperparameter_search_over_model(gridsearch_model_gen, DATASET_INDEX,
    #                                  param_grid={
    #                                      'lstm_cells':[32, 64, 128, 256, 384, 512, 1024],
    #                                      'dense_neurons': [128, 256, 512, 1024, 2048, 4096],
    #                                      'dropout': [0.2, 0.5, 0.8],
    #                                  })



