import json
import os
import traceback

from keras import backend as K
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from keras.layers import Input, PReLU, Dense, LSTM, CuDNNLSTM, concatenate, Activation, GRU, SimpleRNN
from keras.models import Model
from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.generic_utils import load_dataset_at
from utils.keras_utils import train_model, evaluate_model, loss_model
from utils.layer_utils import AttentionLSTM


def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = LSTM(NUM_CELLS)(ip)
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

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_grufcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = GRU(NUM_CELLS)(ip)
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

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_rnnfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = SimpleRNN(NUM_CELLS)(ip)
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

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_cnnfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = Conv1D(NUM_CELLS, 3, padding='same', kernel_initializer='he_uniform')(ip)
    x = Dropout(0.8)(x)
    x = GlobalAveragePooling1D()(x)

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

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_densefcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = Flatten()(ip)
    x = Dense(NUM_CELLS, activation='sigmoid')(x)
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

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_alstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)
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

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_ndlstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = Permute((2, 1))(ip)
    x = CuDNNLSTM(NUM_CELLS)(x)
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

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_ndalstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)
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

    model.summary()

    # add load model code here to fine-tune

    return model


if __name__ == "__main__":

    dataset_map = [('Adiac', 0),
                   ('ArrowHead', 1),
                   ('ChlorineConcentration', 2),
                   ('InsectWingbeatSound', 3),
                   ('Lighting7', 4),
                   ('Wine', 5),
                   ('WordsSynonyms', 6),
                   ('50words', 7),
                   ('Beef', 8),
                   ('DistalPhalanxOutlineAgeGroup', 9),
                   ('DistalPhalanxOutlineCorrect', 10),
                   ('DistalPhalanxTW', 11),
                   ('ECG200', 12),
                   ('ECGFiveDays', 13),
                   ('BeetleFly', 14),
                   ('BirdChicken', 15),
                   ('ItalyPowerDemand', 16),
                   ('SonyAIBORobotSurface', 17),
                   ('SonyAIBORobotSurfaceII', 18),
                   ('MiddlePhalanxOutlineAgeGroup', 19),
                   ('MiddlePhalanxOutlineCorrect', 20),
                   ('MiddlePhalanxTW', 21),
                   ('ProximalPhalanxOutlineAgeGroup', 22),
                   ('ProximalPhalanxOutlineCorrect', 23),
                   ('ProximalPhalanxTW', 24),
                   ('MoteStrain', 25),
                   ('MedicalImages', 26),
                   ('Strawberry', 27),
                   ('ToeSegmentation1', 28),
                   ('Coffee', 29),
                   ('Cricket_X', 30),
                   ('Cricket_Y', 31),
                   ('Cricket_Z', 32),
                   ('uWaveGestureLibrary_X', 33),
                   ('uWaveGestureLibrary_Y', 34),
                   ('uWaveGestureLibrary_Z', 35),
                   ('ToeSegmentation2', 36),
                   ('DiatomSizeReduction', 37),
                   ('car', 38),
                   ('CBF', 39),
                   ('CinC_ECG_torso', 40),
                   ('Computers', 41),
                   ('Earthquakes', 42),
                   ('ECG5000', 43),
                   ('ElectricDevices', 44),
                   ('FaceAll', 45),
                   ('FaceFour', 46),
                   ('FacesUCR', 47),
                   ('Fish', 48),
                   ('FordA', 49),
                   ('FordB', 50),
                   ('Gun_Point', 51),
                   ('Ham', 52),
                   ('HandOutlines', 53),
                   ('Haptics', 54),
                   ('Herring', 55),
                   ('InlineSkate', 56),
                   ('LargeKitchenAppliances', 57),
                   ('Lighting2', 58),
                   ('MALLAT', 59),
                   ('Meat', 60),
                   ('NonInvasiveFatalECG_Thorax1', 61),
                   ('NonInvasiveFatalECG_Thorax2', 62),
                   ('OliveOil', 63),
                   ('OSULeaf', 64),
                   ('PhalangesOutlinesCorrect', 65),
                   ('Phoneme', 66),
                   ('plane', 67),
                   ('RefrigerationDevices', 68),
                   ('ScreenType', 69),
                   ('ShapeletSim', 70),
                   ('ShapesAll', 71),
                   ('SmallKitchenAppliances', 72),
                   ('StarlightCurves', 73),
                   ('SwedishLeaf', 74),
                   ('Symbols', 75),
                   ('synthetic_control', 76),
                   ('Trace', 77),
                   ('Patterns', 78),
                   ('TwoLeadECG', 79),
                   ('UWaveGestureLibraryAll', 80),
                   ('wafer', 81),
                   ('Worms', 82),
                   ('WormsTwoClass', 83),
                   ('yoga', 84),
                   ('ACSF1', 85),
                   ('AllGestureWiimoteX', 86),
                   ('AllGestureWiimoteY', 87),
                   ('AllGestureWiimoteZ', 88),
                   ('BME', 89),
                   ('Chinatown', 90),
                   ('Crop', 91),
                   ('DodgerLoopDay', 92),
                   ('DodgerLoopGame', 93),
                   ('DodgerLoopWeekend', 94),
                   ('EOGHorizontalSignal', 95),
                   ('EOGVerticalSignal', 96),
                   ('EthanolLevel', 97),
                   ('FreezerRegularTrain', 98),
                   ('FreezerSmallTrain', 99),
                   ('Fungi', 100),
                   ('GestureMidAirD1', 101),
                   ('GestureMidAirD2', 102),
                   ('GestureMidAirD3', 103),
                   ('GesturePebbleZ1', 104),
                   ('GesturePebbleZ2', 105),
                   ('GunPointAgeSpan', 106),
                   ('GunPointMaleVersusFemale', 107),
                   ('GunPointOldVersusYoung', 108),
                   ('HouseTwenty', 109),
                   ('InsectEPGRegularTrain', 110),
                   ('InsectEPGSmallTrain', 111),
                   ('MelbournePedestrian', 112),
                   ('MixedShapesRegularTrain', 113),
                   ('MixedShapesSmallTrain', 114),
                   ('PickupGestureWiimoteZ', 115),
                   ('PigAirwayPressure', 116),
                   ('PigArtPressure', 117),
                   ('PigCVP', 118),
                   ('PLAID', 119),
                   ('PowerCons', 120),
                   ('Rock', 121),
                   ('SemgHandGenderCh2', 122),
                   ('SemgHandMovementCh2', 123),
                   ('SemgHandSubjectCh2', 124),
                   ('ShakeGestureWiimoteZ', 125),
                   ('SmoothSubspace', 126),
                   ('UMD', 127)
                   ]

    print("Num datasets : ", len(dataset_map))
    print()

    base_log_name = '%s_%d_cell_search.csv'
    base_weights_dir = '%s_%d_cells_weights/'

    MODELS = [
        ('lstmfcn', generate_lstmfcn),
        # ('alstmfcn', generate_alstmfcn),
        # ('ndlstmfcn', generate_ndlstmfcn),
        # ('grufcn', generate_grufcn),
        # ('rnnfcn', generate_rnnfcn),
        # ('densefcn', generate_densefcn),
    ]

    # Number of cells
    CELLS = [8, 64, 128]

    for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
        for dname, did in dataset_map:

            current_loss = 1e10

            MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[did]
            NB_CLASS = NB_CLASSES_LIST[did]
            finalcell = None

            successes = []
            failures = []

            for cell in CELLS:

                if not os.path.exists(base_log_name % (MODEL_NAME, cell)):
                    file = open(base_log_name % (MODEL_NAME, cell), 'w')
                    file.write('%s,%s,%s,%s\n' % ('dataset_id', 'dataset_name', 'dataset_name_', 'test_accuracy'))
                    file.close()

                # release GPU Memory
                K.clear_session()
                weights_dir = base_weights_dir % (MODEL_NAME, cell)

                if not os.path.exists('weights/' + weights_dir):
                    os.makedirs('weights/' + weights_dir)

                dataset_name_ = weights_dir + dname

                model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell)

                print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)

                train_model(model, did, dataset_name_, epochs=2000, batch_size=128, normalize_timeseries=True)

                model_loss = loss_model(model, did, dataset_name_, batch_size=128, normalize_timeseries=True)

                if (model_loss < current_loss):
                    finalcell = cell
                    current_loss = model_loss

            print('Final Cell Selected:', finalcell)
            model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, finalcell)
            weights_dir = base_weights_dir % (MODEL_NAME, finalcell)
            dataset_name_ = weights_dir + dname

            print(dataset_name_)
            acc = evaluate_model(model, did, dataset_name_, batch_size=128, normalize_timeseries=True)

            s = "%d,%s,%s,%0.6f\n" % (did, dname, dataset_name_, acc)

            file = open(base_log_name % (MODEL_NAME, finalcell), 'a+')
            file.write(s)
            file.flush()

            successes.append(s)

            file.close()
