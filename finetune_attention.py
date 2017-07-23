from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualise_attention, visualize_cam
from utils.layers import AttentionLSTM

from keras import backend as K

import os
import shutil

ATTENTION_CONCAT_AXIS = 1

def generate_model(lstm_size):
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = LSTM(lstm_size)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

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

    # model.summary()

    return model

def generate_model_2(lstm_size):
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(lstm_size)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

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

    # model.summary()

    # add load model code here to fine-tune

    return model


def generate_model_3(lstm_size):
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = attention_block(ip, id=1)
    x = concatenate([ip, x], axis=ATTENTION_CONCAT_AXIS)

    x = LSTM(lstm_size)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

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

    # model.summary()

    # add load model code here to fine-tune

    return model


def attention_block(inputs, id):
    # input shape: (batch_size, time_step, input_dim)
    # input shape: (batch_size, max_sequence_length, lstm_output_dim)
    x = Dense(MAX_SEQUENCE_LENGTH, activation='softmax', name='attention_dense_%d' % id)(inputs)
    x = multiply([inputs, x])
    return x


if __name__ == "__main__":
    if not os.path.exists('weights/finetune_attention/'):
        os.makedirs('weights/finetune_attention/')

    dataset_name_prefix = [
        # Add the 17 datasets here
    ]

    idsetnumber = [
        # Add the 17 dataset ids here
    ]

    lstm_sizes = [
        # Add the 17 lstm sizes corresponding to the best scores obtained on the 17 models
    ]

    # 5 learning rates for 5 tests
    learning_rates = [1e-3, 5e-4, 2e-4, 1e-4, 1e-4]
    batch_sizes = [128, 128, 64, 64, 32]

    for dataset_name, dataset_id, lstm_size in zip(dataset_name_prefix, idsetnumber, lstm_sizes):
        for learning_rate, batch_size in zip(learning_rates, batch_sizes):
            DATASET_INDEX = dataset_id
            MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]

            global NB_CLASS
            NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

            global TRAINABLE
            TRAINABLE = True

            K.clear_session()

            model = generate_model_2(lstm_size)

            param_count = model.count_params()

            dataset_prefix = 'finetune_attention/' + dataset_name + "_lstm_%d" % \
                                                                    (lstm_size)

            prev_best_accuracy = -1
            weights_path = dataset_prefix + "_weights.h5"
            weights_copy_path = dataset_id + "_weights_best.h5"

            if os.path.exists(weights_path):
                model.load_weights(weights_path)
                print("Model weights loaded for dataset %d (%s)" % (dataset_id, dataset_name))

                prev_best_accuracy = evaluate_model(model, DATASET_INDEX, dataset_prefix,
                                                    batch_size=128)

                # preserve old best weights in case next run yields poorer performance than now
                shutil.copy(weights_path, weights_copy_path)

            train_model(model, DATASET_INDEX,
                        dataset_prefix=dataset_prefix,
                        epochs=3000, batch_size=batch_size, learning_rate=learning_rate)

            current_accuracy = evaluate_model(model, DATASET_INDEX, dataset_prefix,
                                              batch_size=128)

            if current_accuracy < prev_best_accuracy:
                os.remove(weights_path) # delete failed fine tuning weights
                os.rename(weights_copy_path, weights_path) # revert to previous best weights


