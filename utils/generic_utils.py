import numpy as np
import pandas as pd
import os

from utils.constants import TRAIN_FILES, TEST_FILES


def load_dataset_at(index) -> (np.array, np.array):
    assert index < len(TRAIN_FILES), "Index invalid. Could not load dataset at %d" % index
    print("Loading train / test dataset : ", TRAIN_FILES[index], TEST_FILES[index])

    if os.path.exists(TRAIN_FILES[index]):
        df = pd.read_csv(TRAIN_FILES[index], header=None, encoding='latin-1')
    elif os.path.exists(TRAIN_FILES[index][1:]):
        df = pd.read_csv(TRAIN_FILES[index][1:], header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (TRAIN_FILES[index]))

    is_timeseries = False if TRAIN_FILES[index][-3:] == 'csv' else True

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    if not is_timeseries:
        data_idx = df.columns[1:]
        min_val = min(df.loc[:, data_idx].min())
        if min_val == 0:
            df.loc[:, data_idx] += 1

    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # cast all data into integer (int32)
    if not is_timeseries:
        df[df.columns] = df[df.columns].astype(np.int32)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    labels = df[[0]]
    labels = labels - labels.min()

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_train = df.values
    y_train = labels.values

    if is_timeseries:
        X_train = X_train[:, np.newaxis, :]

    print("Finished loading train dataset..")

    if os.path.exists(TEST_FILES[index]):
        df = pd.read_csv(TEST_FILES[index], header=None, encoding='latin-1')
    elif os.path.exists(TEST_FILES[index][1:]):
        df = pd.read_csv(TEST_FILES[index][1:], header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (TEST_FILES[index]))

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    if not is_timeseries:
        data_idx = df.columns[1:]
        min_val = min(df.loc[:, data_idx].min())
        if min_val == 0:
            df.loc[:, data_idx] += 1

    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # cast all data into integer (int32)
    if not is_timeseries:
        df[df.columns] = df[df.columns].astype(np.int32)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    labels = df[[0]]
    labels = labels - labels.min()

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_test = df.values
    y_test = labels.values

    if is_timeseries:
        X_test = X_test[:, np.newaxis, :]

    print("Finished loading test dataset..")

    return X_train, y_train, X_test, y_test, is_timeseries


def calculate_dataset_metrics(X_train):
    is_timeseries = len(X_train.shape) == 3
    if is_timeseries:
        # timeseries dataset
        max_sequence_length = X_train.shape[-1]
        max_nb_words = None
    else:
        # transformed dataset
        max_sequence_length = X_train.shape[-1]
        max_nb_words = np.amax(X_train) + 1

    return max_nb_words, max_sequence_length


if __name__ == "__main__":
    word_list = []
    seq_len_list = []
    classes = []

    for index in range(4, 5):
        x, y, x_test, y_test, is_timeseries = load_dataset_at(index)
        nb_words, seq_len = calculate_dataset_metrics(x)
        print("-" * 80)
        print("Dataset : ", index + 1)
        print("Train :: X shape : ", x.shape, "Y shape : ", y.shape, "Nb classes : ", len(np.unique(y)))
        print("Test :: X shape : ", x_test.shape, "Y shape : ", y_test.shape, "Nb classes : ", len(np.unique(y)))
        print("Classes : ", np.unique(y))
        print()

        word_list.append(nb_words)
        seq_len_list.append(seq_len)
        classes.append(len(np.unique(y)))

    print("Word List : ", word_list)
    print("Sequence length list : ", seq_len_list)
    print("Max number of classes : ", classes)