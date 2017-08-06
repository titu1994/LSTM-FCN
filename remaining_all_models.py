from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualise_attention, visualize_cam
from utils.layer_utils import AttentionLSTM

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
    from keras import backend as K
    import os
    import csv

    f = open('utils/comparison_results_attentionlstm_remaining.csv', mode='w', newline='')
    csvwriter = csv.writer(f)

    csvwriter.writerow(['dataset_id', 'lstm_size', 'model_type', 'param_count', 'accuracy'])

    if not os.path.exists("weights/size_comparison/"):
        os.makedirs('weights/size_comparison/')

    dataset_name_prefix = [
        "adiac",
        "chlorine_concentration",
        "insect_wingbeat_sound",
        "lighting7", ######this had a cutoff, need to redo it again
        "wine",
        "words_synonyms",
        "fifty_words",
        "distal_phalanx_outline_age_group",
        "distal_phalanx_outline_correct",
        "distal_phalanx_tw",
        "ecg_200",
        "ecg_five_days",
        "italy_power_demand",
        "middle_phalanx_outline_correct",
        "middle_phalanx_tw",
        "proximal_phalanx_outline_age_group",
        "proximal_phalanx_tw",
        "medical_images",
        "strawberry",
        "coffee",
        "cricket_Z",
        "u_wave_gesture_library_x",
        "u_wave_gesture_library_Z",
        "car",
        "cbf",
        "cin_c_ecg_torso",
        "computers",
        "earthquakes",
        "ecg_5000",
        "face_all",
        "faces_ucr",
        "fish",
        "ford_a",
        "hand_outlines",
        "haptics",
        "herring",
        "inline_skate",
        "large_kitchen_appliances",
        "lighting2",
        "mallat",
        "meat",
        "non_invasive_fatal_ecg_thorax1",
        "non_invasive_fatal_ecg_thorax2",
        "osu_leaf",
        "phalanges_outlines_correct",
        "phoneme",
        "plane",
        "refrigeration_devices",
        "screen_type",
        "shapes_all",
        "small_kitchen_appliances",
        "starlight_curves",
        "swedish_leaf",
        "trace",
        "u_wave_gesture_library_all",
        "wafer",
        "worms_two_class"

    ]

    idsetnumber = [
        0,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        16,
        20,
        21,
        22,
        24,
        26,
        27,
        29,
        32,
        33,
        35,
        38,
        39,
        40,
        41,
        42,
        43,
        45,
        47,
        48,
        49,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        64,
        65,
        66,
        67,
        68,
        69,
        71,
        72,
        73,
        74,
        77,
        80,
        81,
        83

    ]


    lstm_sizes = [8, 64, 128]
    model_types = ['attention']

    for dataset_name, dataset_id in zip(dataset_name_prefix, idsetnumber):
        for model_type in model_types:
            for lstm_size in lstm_sizes:
                DATASET_INDEX = dataset_id
                MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]

                global NB_CLASS
                NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

                global TRAINABLE
                TRAINABLE = True

                K.clear_session()

                if model_type == 'attention':
                    model = generate_model_2(lstm_size)
                else:
                    model = generate_model(lstm_size)

                param_count = model.count_params()

                dataset_prefix = 'size_comparison/' + dataset_name + "_lstm_%d_model_type_%s" % \
                                                                     (lstm_size, model_type)

                train_model(model, DATASET_INDEX,
                            dataset_prefix=dataset_prefix,
                            epochs=3000, batch_size=128,)

                accuracy = evaluate_model(model, DATASET_INDEX, dataset_prefix,
                                          batch_size=128)

                row = [DATASET_INDEX, lstm_size, model_type, param_count, accuracy]

                csvwriter.writerow(row)

    f.close()

