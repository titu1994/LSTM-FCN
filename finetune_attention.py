from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualise_attention, visualize_cam
from utils.layer_utils import AttentionLSTM

from keras import backend as K

import os
import csv
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

    f = open('utils/attention_finetuning_noattention-remainingsets.csv', mode='w', newline='')
    csvwriter = csv.writer(f)

    if not os.path.exists('weights/finetune_attention/'):
        os.makedirs('weights/finetune_attention/')

    csvwriter.writerow(['dataset_id', 'accuracy'])

    dataset_name_prefix = [
        # #"cbf",
        # #"chlorine_concentration",
        # #"cricket_x",
        # ##"ecg_200",
        # #"faces_ucr",
        # #"ford_a",
        # #"fordb",
        # ##"lighting7",
        # ##"meat",
        # #"mote_strain",
        # #"screen_type",
        # #"shapelet_sim",
        # ##"small_kitchen_appliances",
        # ##"sony_aibo_robot_surface",
        # #"synthetic_control",
        # ##"two_lead_ecg",
        # #"wafer"
        # # Add the 17 datasets here
        # #"cin_c_ecg_torso",
        # #"cricket_z",
        # #"ecg_five_days",
        # "electric_devices",
        # #"face_four",
        # #"fish",
        # "hand_outlines",
        # #"inline_skate",
        # #"italy_power_demand",
        # #"lighting2",
        # "starlight_curves",
        # #"toe_segmentation2",
        # "two_patterns",
        # #"u_wave_gesture_library_all",
        # #"words_synonyms",
        # #"worms",
        # #"worms_two_class"
        "adiac",
        "chlorine_concentration",
        "insect_wingbeat_sound",
        "lighting7",  ######this had a cutoff, need to redo it again
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
        # #39,
        # #2,
        # #30,
        # #12,
        # #47,
        # #49,
        # #50,
        # #4,
        # #60,
        # #25,
        # #69,
        # #70,
        # #72,
        # ##17,
        # #76,
        # #79,
        # #81,
        # #40,
        # #31,
        # #13,
        # 44,
        # #46,
        # #48,
        # 53,
        # #56,
        # #16,
        # #58,
        # 73,
        # #36,
        # 78,
        # #80,
        # #6,
        # #82,
        # #83
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

    lstm_sizes = [  # 8,
        # # 128,
        # # 8,
        # ##8,
        # # 64,
        # # 64,
        # # 8,
        # ##8,
        # # 64,
        # # 64,
        # # 8,
        # # 64,
        # ##8,
        # ##8,
        # # 64,
        # ##64,
        # # 8,
        # # Add the 17 lstm sizes corresponding to the best scores obtained on the 17 models
        # #64,
        # #64,
        # #64,
        # 64,
        # #8,
        # #128,
        # 128,
        # #8,
        # #64,
        # #8,
        # 128,
        # #64,
        # 64,
        # #128,
        # #64,
        # #8,
        # #8,
        8,
        8,
        64,
        8,
        64,
        128,
        128,
        64,
        64,
        64,
        128,
        8,
        8,
        64,
        64,
        64,
        64,
        8,
        128,
        128,
        8,
        128,
        128,
        64,
        8,
        128,
        8,
        8,
        128,
        8,
        128,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        128,
        64,
        128,
        8,
        8,

    ]

    # 5 learning rates for 5 tests
    # learning_rates = [1e-3, 5e-4, 2e-4, 1e-4, 1e-4]
    # batch_sizes = [128, 128, 64, 64, 32]

    # learning_rates = [1e-5, 5e-5, 2e-5, 1e-5, 1e-5]
    # batch_sizes = [128, 128, 64, 64, 32]

    learning_rates = [1e-3]
    batch_sizes = [128]

    for dataset_name, dataset_id, lstm_size in zip(dataset_name_prefix, idsetnumber, lstm_sizes):
        best_score = []

        for learning_rate, batch_size in zip(learning_rates, batch_sizes):
            DATASET_INDEX = dataset_id
            MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]

            global NB_CLASS
            NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

            global TRAINABLE
            TRAINABLE = True

            K.clear_session()

            model = generate_model(lstm_size)

            param_count = model.count_params()

            dataset_prefix = 'no_attention_no_finetuning/' + dataset_name

            prev_best_accuracy = -1
            weights_path = "weights/" + dataset_prefix + "_weights.h5"
            weights_copy_path = "weights/" + dataset_prefix + "_weights_best.h5"

            if os.path.exists(weights_path):
                #model.load_weights(weights_path)
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

            best_score.append(current_accuracy)

        csvwriter.writerow([dataset_id, str(max(best_score))])


