import numpy as np
import pandas as pd
import os

from utils.constants import TRAIN_FILES, TEST_FILES, MAX_SEQUENCE_LENGTH_LIST


def load_dataset_at(index, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    assert index < len(TRAIN_FILES), "Index invalid. Could not load dataset at %d" % index
    if verbose: print("Loading train / test dataset : ", TRAIN_FILES[index], TEST_FILES[index])

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
        # scale the values
        if normalize_timeseries:
            X_train = (X_train - X_train.mean()) / (X_train.std())

    if verbose: print("Finished loading train dataset..")

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
        # scale the values
        if normalize_timeseries:
            X_test = (X_test - X_test.mean()) / (X_test.std())

    if verbose: print("Finished loading test dataset..")

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


def plot_dataset(dataset_id, seed=None, limit=None, cutoff=None,
                 normalize_timeseries=False, pass_data=None):
    import matplotlib.pylab as plt
    np.random.seed(seed)

    if pass_data is None:
        X_train, _, X_test, _, is_timeseries = load_dataset_at(dataset_id,
                                                               normalize_timeseries=normalize_timeseries)

        if not is_timeseries:
            print("Can plot time series input data only!\n"
                  "Continuing without plot!")
            return

        max_nb_words, sequence_length = calculate_dataset_metrics(X_train)

        if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
            if cutoff is None:
                choice = cutoff_choice(dataset_id, sequence_length)
            else:
                assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
                choice = cutoff

            if choice not in ['pre', 'post']:
                return
            else:
                X_train, X_test = X_test(X_train, X_test, choice, dataset_id, sequence_length)

        X_train_attention = None
        X_test_attention = None

    else:
        X_train, X_test, X_train_attention, X_test_attention = pass_data

    if limit is None:
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
    else:
        train_size = limit
        test_size = limit

    train_idx = np.random.randint(0, X_train.shape[0], size=train_size)
    X_train = X_train[train_idx, 0, :]
    X_train = X_train.transpose((1, 0))

    if X_train_attention is not None:
        X_train_attention = X_train_attention[train_idx, 0, :]
        X_train_attention = X_train_attention.transpose((1, 0))

    test_idx = np.random.randint(0, X_test.shape[0], size=test_size)
    X_test = X_test[test_idx, 0, :]
    X_test = X_test.transpose((1, 0))

    if X_test_attention is not None:
        X_test_attention = X_test_attention[test_idx, 0, :]
        X_test_attention = X_test_attention.transpose((1, 0))

    train_df = pd.DataFrame(X_train,
                            index=range(X_train.shape[0]),
                            columns=range(X_train.shape[1]))

    test_df = pd.DataFrame(X_test,
                           index=range(X_test.shape[0]),
                           columns=range(X_test.shape[1]))

    if pass_data is not None:
        rows = 2
        cols = 2
    else:
        rows = 1
        cols = 2

    fig, axs = plt.subplots(rows, cols)

    train_df.plot(title='Train dataset',
                  subplots=False,
                  legend=None,
                  ax=axs[0][0])

    test_df.plot(title='Test dataset',
                 subplots=False,
                 legend=None,
                 ax=axs[1][0])

    if pass_data is not None and X_train_attention is not None:
        print(X_train_attention.shape)
        train_attention_df = pd.DataFrame(X_train_attention,
                            index=range(X_train_attention.shape[0]),
                            columns=range(X_train_attention.shape[1]))

        train_attention_df.plot(title='Train Attention Sequence',
                      subplots=False,
                      legend=None,
                      ax=axs[0][1])

    if pass_data is not None and X_test_attention is not None:
        print(X_test_attention.shape)
        test_df = pd.DataFrame(X_test_attention,
                               index=range(X_test_attention.shape[0]),
                               columns=range(X_test_attention.shape[1]))

        test_df.plot(title='Test Attention Sequence',
                     subplots=False,
                     legend=None,
                     ax=axs[1][1])

    plt.show()


def cutoff_choice(dataset_id, sequence_length):
    print("Original sequence length was :", sequence_length, "New sequence Length will be : ",
          MAX_SEQUENCE_LENGTH_LIST[dataset_id])
    choice = input('Options : \n'
                   '`pre` - cut the sequence from the beginning\n'
                   '`post`- cut the sequence from the end\n'
                   '`anything else` - stop execution\n'
                   'To automate choice: add flag `cutoff` = choice as above\n'
                   'Choice = ')

    choice = str(choice).lower()
    return choice


def cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length):
    assert MAX_SEQUENCE_LENGTH_LIST[dataset_id] < sequence_length, "If sequence is to be cut, max sequence" \
                                                                   "length must be less than original sequence length."
    cutoff = sequence_length - MAX_SEQUENCE_LENGTH_LIST[dataset_id]
    if choice == 'pre':
        if X_train is not None:
            X_train = X_train[:, :, cutoff:]
        if X_test is not None:
            X_test = X_test[:, :, cutoff:]
    else:
        if X_train is not None:
            X_train = X_train[:, :, :-cutoff]
        if X_test is not None:
            X_test = X_test[:, :, :-cutoff]
    print("New sequence length :", MAX_SEQUENCE_LENGTH_LIST[dataset_id])
    return X_train, X_test


if __name__ == "__main__":
    # word_list = []
    # seq_len_list = []
    # classes = []
    #
    # for index in range(6, 9):
    #     x, y, x_test, y_test, is_timeseries = load_dataset_at(index)
    #     nb_words, seq_len = calculate_dataset_metrics(x)
    #     print("-" * 80)
    #     print("Dataset : ", index + 1)
    #     print("Train :: X shape : ", x.shape, "Y shape : ", y.shape, "Nb classes : ", len(np.unique(y)))
    #     print("Test :: X shape : ", x_test.shape, "Y shape : ", y_test.shape, "Nb classes : ", len(np.unique(y)))
    #     print("Classes : ", np.unique(y))
    #     print()
    #
    #     word_list.append(nb_words)
    #     seq_len_list.append(seq_len)
    #     classes.append(len(np.unique(y)))
    #
    # print("Word List : ", word_list)
    # print("Sequence length list : ", seq_len_list)
    # print("Max number of classes : ", classes)

    #print()
    plot_dataset(dataset_id=28, seed=1, limit=None, cutoff=None, normalize_timeseries=True)