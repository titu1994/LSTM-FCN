import os

from keras import backend as K
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from keras.layers import Input, Dense, LSTM, CuDNNLSTM, concatenate, Activation, GRU, SimpleRNN
from keras.models import Model

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model
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

    base_log_name = '%s_%d_cells_new_datasets.csv'
    base_weights_dir = '%s_%d_cells_weights/'

    MODELS = [
        ('lstmfcn', generate_lstmfcn),
        ('alstmfcn', generate_alstmfcn),
    ]

    # Number of cells
    CELLS = [8, 64, 128]

    # Normalization scheme
    # Normalize = False means no normalization will be done
    # Normalize = True / 1 means sample wise z-normalization
    # Normalize = 2 means dataset normalization.
    normalize_dataset = True

    for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
        for cell in CELLS:
            successes = []
            failures = []

            if not os.path.exists(base_log_name % (MODEL_NAME, cell)):
                file = open(base_log_name % (MODEL_NAME, cell), 'w')
                file.write('%s,%s,%s,%s\n' % ('dataset_id', 'dataset_name', 'dataset_name_', 'test_accuracy'))
                file.close()

            for dname, did in dataset_map:

                MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[did]
                NB_CLASS = NB_CLASSES_LIST[did]

                # release GPU Memory
                K.clear_session()

                file = open(base_log_name % (MODEL_NAME, cell), 'a+')

                weights_dir = base_weights_dir % (MODEL_NAME, cell)

                if not os.path.exists('weights/' + weights_dir):
                    os.makedirs('weights/' + weights_dir)

                dataset_name_ = weights_dir + dname

                # try:
                model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell)

                print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)

                # comment out the training code to only evaluate !
                train_model(model, did, dataset_name_, epochs=2000, batch_size=128,
                            normalize_timeseries=normalize_dataset)

                acc = evaluate_model(model, did, dataset_name_, batch_size=128,
                                     normalize_timeseries=normalize_dataset)

                s = "%d,%s,%s,%0.6f\n" % (did, dname, dataset_name_, acc)

                file.write(s)
                file.flush()

                successes.append(s)

                # except Exception as e:
                #     traceback.print_exc()
                #
                #     s = "%d,%s,%s,%s\n" % (did, dname, dataset_name_, 0.0)
                #     failures.append(s)
                #
                #     print()

                file.close()

            print('\n\n')
            print('*' * 20, "Successes", '*' * 20)
            print()

            for line in successes:
                print(line)

            print('\n\n')
            print('*' * 20, "Failures", '*' * 20)
            print()

            for line in failures:
                print(line)
