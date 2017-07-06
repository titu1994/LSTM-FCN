from keras.models import Model
from keras.layers import Input, PReLU, Dense,Dropout, LSTM, Bidirectional, multiply, concatenate
from keras import backend as K

from utils.constants import MAX_NB_WORDS_LIST, MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualise_attention, visualize_cam

DATASET_INDEX = 10

MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
MAX_NB_WORDS = MAX_NB_WORDS_LIST[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

ATTENTION_CONCAT_AXIS = 1 # 1 = temporal, -1 = spatial
TRAINABLE = True


def generate_model():

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH,))

    a = attention_block(ip, id=1)
    x = concatenate([ip, a], axis=ATTENTION_CONCAT_AXIS)

    x = Bidirectional(LSTM(512, trainable=TRAINABLE))(x)

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

    #train_model(model, DATASET_INDEX, dataset_prefix='phalanx_outline_timesequence', epochs=150, batch_size=128,
    #            val_subset=276)

    evaluate_model(model, DATASET_INDEX, dataset_prefix='phalanx_outline_timesequence', batch_size=128,
                  test_data_subset=276)

    visualise_attention(model, DATASET_INDEX, 'phalanx_outline_timesequence', layer_name='attention_dense_1')

    visualize_cam(model, DATASET_INDEX, dataset_prefix='phalanx_outline_timesequence', class_id=0)
