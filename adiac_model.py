from keras.models import Model
from keras.layers import Input, PReLU, Dense, Dropout, LSTM, Bidirectional, multiply, concatenate
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute

from utils.constants import MAX_NB_WORDS_LIST, MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualise_attention

DATASET_INDEX = 0

MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
MAX_NB_WORDS = MAX_NB_WORDS_LIST[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

ATTENTION_CONCAT_AXIS = 1 # 1 = temporal, -1 = spatial
TRAINABLE = True

def generate_model():
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = attention_block(ip, id=1)
    x = concatenate([ip, x], axis=ATTENTION_CONCAT_AXIS)

    x = Bidirectional(LSTM(128, trainable=TRAINABLE))(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 9, padding='same')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    y = Conv1D(256, 5, padding='same')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    y = Conv1D(256, 3, padding='same')(y)
    y = BatchNormalization()(y)
    y = PReLU()(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    for layer in model.layers[:-4]:
        set_trainable(layer, TRAINABLE)

    model.summary()

    from keras.utils.vis_utils import plot_model
    plot_model(model, 'utils/modelv3.png', show_shapes=True)

    return model


def attention_block(inputs, id):
    # input shape: (batch_size, time_step, input_dim)
    # input shape: (batch_size, max_sequence_length, lstm_output_dim)
    x = Dense(MAX_SEQUENCE_LENGTH, activation='softmax', name='attention_dense_%d' % id)(inputs)
    x = multiply([inputs, x])
    return x


if __name__ == "__main__":
    model = generate_model()

    train_model(model, DATASET_INDEX, dataset_prefix='adiac', epochs=2000, batch_size=128)

    evaluate_model(model, DATASET_INDEX, dataset_prefix='adiac', batch_size=128)

    visualise_attention(model, DATASET_INDEX, dataset_prefix='adiac', layer_name='attention_dense_1')