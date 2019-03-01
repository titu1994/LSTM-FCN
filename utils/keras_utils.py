import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.transform import resize

import warnings

from keras.models import Model
from keras.layers import Permute
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, \
    cutoff_sequence, plot_dataset
from utils.constants import MAX_SEQUENCE_LENGTH_LIST, TRAIN_FILES

mpl.style.use('seaborn-paper')
warnings.simplefilter('ignore', category=DeprecationWarning)


if not os.path.exists('weights/'):
    os.makedirs('weights/')


def train_model(model: Model, dataset_id, dataset_prefix, epochs=50, batch_size=128, val_subset=None,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-3):
    """
    Trains a provided Model, given a dataset id.

    Args:
        model: A Keras Model.
        dataset_id: Integer id representing the dataset index containd in
            `utils/constants.py`.
        dataset_prefix: Name of the dataset. Used for weight saving.
        epochs: Number of epochs to train.
        batch_size: Size of each batch for training.
        val_subset: Optional integer id to subset the test set. To be used if
            the test set evaluation time significantly surpasses training time
            per epoch.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        learning_rate: Initial learning rate.
    """
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      normalize_timeseries=normalize_timeseries)
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
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    if not is_timeseries:
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH_LIST[dataset_id], padding='post', truncating='post')
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH_LIST[dataset_id], padding='post', truncating='post')

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                                 np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)

    path_splits = os.path.split(dataset_prefix)
    if len(path_splits) > 1:
        base_path = os.path.join('weights', *path_splits)

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        base_path = os.path.join(base_path, path_splits[-1])

    else:
        all_weights_path = os.path.join('weights', dataset_prefix)

        if not os.path.exists(all_weights_path):
            os.makedirs(all_weights_path)

    model_checkpoint = ModelCheckpoint("./weights/%s_weights.h5" % dataset_prefix, verbose=1,
                                       monitor='loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)

    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))


def evaluate_model(model: Model, dataset_id, dataset_prefix, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    """
    Evaluates a given Keras Model on the provided dataset.

    Args:
        model: A Keras Model.
        dataset_id: Integer id representing the dataset index containd in
            `utils/constants.py`.
        dataset_prefix: Name of the dataset. Used for weight saving.
        batch_size: Size of each batch for evaluation.
        test_data_subset: Optional integer id to subset the test set. To be used if
            the test set evaluation time is significantly.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.

    Returns:
        The test set accuracy of the model.
    """
    _, _, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                          normalize_timeseries=normalize_timeseries)
    max_nb_words, sequence_length = calculate_dataset_metrics(X_test)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, dataset_id, sequence_length)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH_LIST[dataset_id], padding='post', truncating='post')
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)
    print("Weights loaded from ", "./weights/%s_weights.h5" % dataset_prefix)

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    print()
    print("Final Accuracy : ", accuracy)

    return accuracy


def set_trainable(layer, value):
    """
    Sets the layers of the Model to be trainable or not.

    Args:
        layer: can be a single Layer of a Model, or an entire Model.
        value: True or False.
    """
    layer.trainable = value

    # case: container
    if hasattr(layer, 'layers'):
        for l in layer.layers:
            set_trainable(l, value)

    # case: wrapper (which is a case not covered by the PR)
    if hasattr(layer, 'layer'):
        set_trainable(layer.layer, value)


def build_function(model, layer_names=None, outputs=None):
    """
    Builds a Keras Function which retrieves the output of a Layer.

    Args:
        model: Keras Model.
        layer_names: Name of the layer whose output is required.
        outputs: Output tensors.

    Returns:
        List of Keras Functions.
    """
    inp = model.input

    if layer_names is not None and (type(layer_names) != list and type(layer_names) != tuple):
        layer_names = [layer_names]

    if outputs is None:
        if layer_names is None:
            outputs = [layer.output for layer in model.layers]  # all layer outputs
        else:
            outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    else:
        outputs = outputs

    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    return funcs


def get_outputs(model, inputs, eval_functions, verbose=False):
    """
    Gets the outputs of the Keras model.

    Args:
        model: Unused.
        inputs: Input numpy arrays.
        eval_functions: Keras functions for evaluation.
        verbose: Whether to print evaluation metrics.

    Returns:
        List of outputs of the Keras Model.
    """
    if verbose: print('----- activations -----')
    outputs = []
    layer_outputs = [func([inputs, 1.])[0] for func in eval_functions]
    for layer_activations in layer_outputs:
        outputs.append(layer_activations)
    return outputs


def visualize_context_vector(model: Model, dataset_id, dataset_prefix, cutoff=None, limit=None,
                             normalize_timeseries=False, visualize_sequence=True, visualize_classwise=False):
    """
    Visualize the Context Vector of the Attention LSTM.

    Args:
        model: an Attention LSTM-FCN Model.
        dataset_id: Integer id representing the dataset index containd in
            `utils/constants.py`.
        dataset_prefix: Name of the dataset. Used for weight saving.
        batch_size: Size of each batch for evaluation.
        test_data_subset: Optional integer id to subset the test set. To be used if
            the test set evaluation time is significantly.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        limit: Number of samples to be visualized in one plot.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        visualize_sequence: Bool flag, whetehr to visualize the sequence attended to
            by the Context Vector or just the Context Vector itself.
        visualize_classwise: Bool flag. Wheter to visualize the samples
            seperated by class. When doing so, `limit` is multiplied by
            the number of classes so it is better to set `limit` to 1 in
            such cases.
    """

    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      normalize_timeseries=normalize_timeseries)
    _, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    attn_lstm_layer = [(i, layer) for (i, layer) in enumerate(model.layers)
                       if layer.__class__.__name__ == 'AttentionLSTM']

    if len(attn_lstm_layer) == 0:
        raise AttributeError('Provided model does not have an Attention layer')
    else:
        i, attn_lstm_layer = attn_lstm_layer[0]  # use first attention lstm layer only

    attn_lstm_layer.return_attention = True

    model.layers[i] = attn_lstm_layer
    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    attention_output = model.layers[i].call(model.input)

    eval_functions = build_function(model, attn_lstm_layer.name, outputs=[attention_output])
    train_attention_vectors = []
    test_attention_vectors = []

    output_shape = [X_train.shape[-1], 1, 1]

    for i in range(X_train.shape[0]):
        activations = get_outputs(model,
                                  X_train[i, :, :][np.newaxis, ...],
                                  eval_functions,
                                  verbose=False)[0]

        # print("activations", activations.shape)
        attention_vector = activations.reshape((-1, 1, 1))

        attention_vector = (attention_vector - attention_vector.min()) / (
                attention_vector.max() - attention_vector.min())
        attention_vector = (attention_vector * 2.) - 1.

        attention_vector = resize(attention_vector, output_shape, mode='reflect', anti_aliasing=True)
        attention_vector = attention_vector.reshape([1, -1])
        train_attention_vectors.append(attention_vector)

    for i in range(X_test.shape[0]):
        activations = get_outputs(model,
                                  X_test[i, :, :][np.newaxis, ...],
                                  eval_functions,
                                  verbose=False)[0]

        # print("activations", activations.shape)
        attention_vector = activations.reshape((-1, 1, 1))

        attention_vector = (attention_vector - attention_vector.min()) / (
                attention_vector.max() - attention_vector.min())
        attention_vector = (attention_vector * 2.) - 1.

        attention_vector = resize(attention_vector, output_shape, mode='reflect', anti_aliasing=True)
        attention_vector = attention_vector.reshape([1, -1])
        test_attention_vectors.append(attention_vector)

    train_attention_vectors = np.array(train_attention_vectors)
    test_attention_vectors = np.array(test_attention_vectors)

    print("Train Attention Vectors Shape :", train_attention_vectors.shape)
    print("Test Attentin Vectors Shape :", test_attention_vectors.shape)

    if visualize_sequence:
        # plot input sequence part that is paid attention too in detail
        X_train_attention = train_attention_vectors * X_train
        X_test_attention = test_attention_vectors * X_test

        plot_dataset(dataset_id, seed=1, limit=limit, cutoff=cutoff,
                     normalize_timeseries=normalize_timeseries, plot_data=(X_train, y_train, X_test, y_test,
                                                                           X_train_attention, X_test_attention),
                     type='Context', plot_classwise=visualize_classwise)

    else:
        # plot only attention chart
        choice = np.random.randint(0, train_attention_vectors.shape[0])

        train_df = pd.DataFrame({'attention (%)': train_attention_vectors[choice, 0]},
                                index=range(train_attention_vectors.shape[-1]))

        train_df.plot(kind='bar',
                      title='Attention Mechanism (Train) as '
                            'a function of input'
                            ' dimensions. Class = %d' % (
                                y_train[choice]
                            ))

        plt.show()


def write_context_vector(model: Model, dataset_id, dataset_prefix, cutoff=None, limit=None,
                         normalize_timeseries=False, visualize_sequence=True, visualize_classwise=False):
    """ Same as visualize_context_vector, but writes the context vectors to a file. Unused. """

    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      normalize_timeseries=normalize_timeseries)
    _, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    attn_lstm_layer = [(i, layer) for (i, layer) in enumerate(model.layers)
                       if layer.__class__.__name__ == 'AttentionLSTM']

    if len(attn_lstm_layer) == 0:
        raise AttributeError('Provided model does not have an Attention layer')
    else:
        i, attn_lstm_layer = attn_lstm_layer[0]  # use first attention lstm layer only

    attn_lstm_layer.return_attention = True

    model.layers[i] = attn_lstm_layer
    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    attention_output = model.layers[i].call(model.input)

    eval_functions = build_function(model, attn_lstm_layer.name, outputs=[attention_output])
    train_attention_vectors = []
    test_attention_vectors = []

    if not os.path.exists('lstm_features/'):
        os.makedirs('lstm_features/')

    output_shape = [X_train.shape[-1], 1, 1]

    for i in range(X_train.shape[0]):
        activations = get_outputs(model,
                                  X_train[i, :, :][np.newaxis, ...],
                                  eval_functions,
                                  verbose=False)[0]

        # print("activations", activations.shape)
        attention_vector = activations.reshape((-1, 1, 1))

        attention_vector = (attention_vector - attention_vector.min()) / (
                attention_vector.max() - attention_vector.min())
        attention_vector = (attention_vector * 2.) - 1.

        attention_vector = resize(attention_vector, output_shape, mode='reflect', anti_aliasing=True)
        attention_vector = attention_vector.reshape([1, -1])
        train_attention_vectors.append(attention_vector)

    for i in range(X_test.shape[0]):
        activations = get_outputs(model,
                                  X_test[i, :, :][np.newaxis, ...],
                                  eval_functions,
                                  verbose=False)[0]

        # print("activations", activations.shape)
        attention_vector = activations.reshape((-1, 1, 1))

        attention_vector = (attention_vector - attention_vector.min()) / (
                attention_vector.max() - attention_vector.min())
        attention_vector = (attention_vector * 2.) - 1.

        attention_vector = resize(attention_vector, output_shape, mode='reflect', anti_aliasing=True)
        attention_vector = attention_vector.reshape([1, -1])
        test_attention_vectors.append(attention_vector)

    train_attention_vectors = np.array(train_attention_vectors)
    test_attention_vectors = np.array(test_attention_vectors)

    print("Train Attention Vectors Shape :", train_attention_vectors.shape)
    print("Test Attentin Vectors Shape :", test_attention_vectors.shape)

    if visualize_sequence:
        # plot input sequence part that is paid attention too in detail
        X_train_attention = train_attention_vectors * X_train
        X_test_attention = test_attention_vectors * X_test

        X_train_attention = X_train_attention.squeeze(1)
        X_test_attention = X_test_attention.squeeze(1)

        df = pd.DataFrame(X_test_attention)
        df['label'] = y_test[:, 0]

        df.to_csv('lstm_features/features.csv')

    else:
        # plot only attention chart
        choice = np.random.randint(0, train_attention_vectors.shape[0])

        train_df = pd.DataFrame({'attention (%)': train_attention_vectors[choice, 0]},
                                index=range(train_attention_vectors.shape[-1]))

        train_df.plot(kind='bar',
                      title='Attention Mechanism (Train) as '
                            'a function of input'
                            ' dimensions. Class = %d' % (
                                y_train[choice]
                            ))

        plt.show()


def visualize_cam(model: Model, dataset_id, dataset_prefix, class_id,
                  cutoff=None, normalize_timeseries=False, seed=0):
    """
    Used to visualize the Class Activation Maps of the Keras Model.

    Args:
        model: A Keras Model.
        dataset_id: Integer id representing the dataset index containd in
            `utils/constants.py`.
        dataset_prefix: Name of the dataset. Used for weight saving.
        class_id: Index of the class whose activation is to be visualized.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        seed: Random seed number for Numpy.
    """

    np.random.seed(seed)

    X_train, y_train, _, _, is_timeseries = load_dataset_at(dataset_id,
                                                            normalize_timeseries=normalize_timeseries)
    _, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, _ = cutoff_sequence(X_train, _, choice, dataset_id, sequence_length)

    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    class_weights = model.layers[-1].get_weights()[0]

    conv_layers = [layer for layer in model.layers if layer.__class__.__name__ == 'Conv1D']

    final_conv = conv_layers[-1].name
    final_softmax = model.layers[-1].name
    out_names = [final_conv, final_softmax]

    if class_id > 0:
        class_id = class_id - 1

    y_train_ids = np.where(y_train[:, 0] == class_id)
    sequence_input = X_train[y_train_ids[0], ...]
    choice = np.random.choice(range(len(sequence_input)), 1)
    sequence_input = sequence_input[choice, :, :]

    eval_functions = build_function(model, out_names)
    conv_out, predictions = get_outputs(model, sequence_input, eval_functions)

    conv_out = conv_out[0, :, :]  # (T, C)

    conv_out = (conv_out - conv_out.min(axis=0, keepdims=True)) / \
               (conv_out.max(axis=0, keepdims=True) - conv_out.min(axis=0, keepdims=True))
    conv_out = (conv_out * 2.) - 1.

    conv_out = conv_out.transpose((1, 0))  # (C, T)
    conv_channels = conv_out.shape[0]

    conv_cam = class_weights[:conv_channels, [class_id]] * conv_out
    conv_cam = np.sum(conv_cam, axis=0)

    conv_cam /= conv_cam.max()

    sequence_input = sequence_input.reshape((-1, 1))
    conv_cam = conv_cam.reshape((-1, 1))

    sequence_df = pd.DataFrame(sequence_input,
                               index=range(sequence_input.shape[0]),
                               columns=range(sequence_input.shape[1]))

    conv_cam_df = pd.DataFrame(conv_cam,
                               index=range(conv_cam.shape[0]),
                               columns=[1])

    fig, axs = plt.subplots(2, 1, squeeze=False,
                            figsize=(6, 6))

    class_label = class_id + 1

    sequence_df.plot(title='Sequence (class = %d)' % (class_label),
                     subplots=False,
                     legend=None,
                     ax=axs[0][0])

    conv_cam_df.plot(title='Convolution Class Activation Map (class = %d)' % (class_label),
                     subplots=False,
                     legend=None,
                     ax=axs[1][0])

    plt.show()


def write_cam(model: Model, dataset_id, dataset_prefix,
              cutoff=None, normalize_timeseries=False):
    """ Same as visualize_cam, but writes the result data to a file. """

    _, _, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                          normalize_timeseries=normalize_timeseries)
    _, sequence_length = calculate_dataset_metrics(X_test)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, _ = cutoff_sequence(_, X_test, choice, dataset_id, sequence_length)

    print("Weights path : ", "./weights/%s_weights.h5" % dataset_prefix)
    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    class_weights = model.layers[-1].get_weights()[0]

    conv_layers = [layer for layer in model.layers if layer.__class__.__name__ == 'Conv1D']

    final_conv = conv_layers[-1].name
    final_softmax = model.layers[-1].name
    out_names = [final_conv, final_softmax]

    eval_functions = build_function(model, out_names)

    parts = os.path.split(dataset_prefix)
    if len(parts) > 1:
        basepath = os.path.join('cam_features', os.pathsep.join(parts[:-1]))
        dataset_name = parts[-1]
    else:
        basepath = 'cam_features/'
        dataset_name = dataset_prefix

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    cam_features = []

    for i in range(X_test.shape[0]):
        print("Sample %d running" % (i + 1))
        y_id = y_test[i, 0]
        sequence_input = X_test[[i], :, :]

        conv_out, predictions = get_outputs(model, sequence_input, eval_functions)

        conv_out = conv_out[0, :, :]  # (T, C)

        conv_out = conv_out.transpose((1, 0))  # (C, T)
        conv_channels = conv_out.shape[0]
        conv_cam = class_weights[:conv_channels, [int(y_id)]] * conv_out
        conv_cam = np.mean(conv_cam, axis=0)

        conv_cam = conv_cam.reshape((-1, 1))
        cam_features.append(conv_cam)

    cam_features = np.concatenate(cam_features, -1).transpose()
    print("Num features = ", cam_features.shape)

    conv_cam_df = pd.DataFrame(cam_features)

    conv_cam_df.to_csv(basepath + '/%s_features_mean_unnormalized.csv' % dataset_name,
                       header=False, index=False)


def visualize_filters(model: Model, dataset_id, dataset_prefix,
                      conv_id=0, filter_id=0, seed=0, cutoff=None,
                      normalize_timeseries=False):
    """
    Used to visualize the output filters of a particular convolution layer.

    Args:
        model: A Keras Model.
        dataset_id: Integer id representing the dataset index containd in
            `utils/constants.py`.
        dataset_prefix: Name of the dataset. Used for weight saving.
        conv_id: Convolution layer ID. Can be 0, 1 or 2 for LSTMFCN and
            its univariate variants (as it uses 3 Conv blocks).
        filter_id: ID of the filter that is under observation.
        seed: Numpy random seed.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
    """

    np.random.seed(seed)

    assert conv_id >= 0 and conv_id < 3, "Convolution layer ID must be between 0 and 2"

    X_train, y_train, _, _, is_timeseries = load_dataset_at(dataset_id,
                                                            normalize_timeseries=normalize_timeseries)
    _, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, _ = cutoff_sequence(X_train, _, choice, dataset_id, sequence_length)

    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    conv_layers = [layer for layer in model.layers
                   if layer.__class__.__name__ == 'Conv1D']

    conv_layer = conv_layers[conv_id]
    conv_layer_name = conv_layer.name

    eval_functions = build_function(model, [conv_layer_name])

    save_dir = os.path.split(dataset_prefix)[0]

    if not os.path.exists('cnn_filters/%s' % (save_dir)):
        os.makedirs('cnn_filters/%s' % (save_dir))

    dataset_name = os.path.split(dataset_prefix)[-1]
    if dataset_name is None or len(dataset_name) == 0:
        dataset_name = dataset_prefix

    # Select single datapoint
    sample_index = np.random.randint(0, X_train.shape[0])
    y_id = y_train[sample_index, 0]
    sequence_input = X_train[[sample_index], :, :]

    # Get features of the cnn layer out
    conv_out = get_outputs(model, sequence_input, eval_functions)[0]

    conv_out = conv_out[0, :, :]  # [T, C]

    # select single filter
    assert filter_id > 0 and filter_id < conv_out.shape[-1]
    channel = conv_out[:, filter_id]
    channel = channel.reshape((-1, 1))

    conv_filters = pd.DataFrame(channel)
    conv_filters.to_csv('cnn_filters/%s_features.csv' % (dataset_prefix), header=None, index=False)

    sequence_input = sequence_input[0, :, :].transpose()
    sequence_df = pd.DataFrame(sequence_input,
                               index=range(sequence_input.shape[0]))

    conv_cam_df = pd.DataFrame(conv_filters,
                               index=range(conv_filters.shape[0]))

    fig, axs = plt.subplots(2, 1, squeeze=False,
                            figsize=(6, 6))

    class_label = y_id + 1

    plt.rcParams.update({'font.size': 24})

    sequence_df.plot(title='Dataset %s : Sequence ID = %d (class = %d)' % (dataset_name,
                                                                           sample_index + 1,
                                                                           class_label),
                     subplots=False,
                     legend=None,
                     ax=axs[0][0])

    conv_cam_df.plot(title='Convolution Layer %d Filter ID %d (class = %d)' % (conv_id + 1,
                                                                               filter_id + 1,
                                                                               class_label),
                     subplots=False,
                     legend=None,
                     ax=axs[1][0])

    # Formatting
    plt.xlabel('Timesteps', axes=axs[0][0])

    axs[0][0].set_ylabel('Value')
    axs[1][0].set_ylabel('Value')

    def mjrFormatter(x, pos):
        return '{:.2f}'.format(x)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(mjrFormatter))

    plt.show()


def extract_features(model: Model, dataset_id, dataset_prefix,
                     layer_name, cutoff=None, normalize_timeseries=False):
    """ Same as visualize_features, but saves them to a file instead. """

    layer_name = layer_name.lower()
    assert layer_name in ['cnn', 'lstm', 'lstmfcn']

    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id,
                                                                      normalize_timeseries=normalize_timeseries)
    _, sequence_length = calculate_dataset_metrics(X_train)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    conv_layers = [layer for layer in model.layers
                   if layer.__class__.__name__ == 'Conv1D']

    lstm_layers = [layer for layer in model.layers
                   if layer.__class__.__name__ == 'LSTM' or
                   layer.__class__.__name__ == 'AttentionLSTM']

    lstmfcn_layer = model.layers[-2]

    if layer_name == 'cnn':
        feature_layer = conv_layers[-1]
    elif layer_name == 'lstm':
        feature_layer = lstm_layers[-1]
    else:
        feature_layer = lstmfcn_layer

    dataset_name = os.path.split(dataset_prefix)[-1]

    if dataset_name is None or len(dataset_name) == 0:
        dataset_name = dataset_prefix

    save_dir = 'layer_features/%s/' % dataset_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    extraction_model = Model(model.input, feature_layer.output)

    train_features = extraction_model.predict(X_train, batch_size=128)
    test_features = extraction_model.predict(X_test, batch_size=128)

    shape = train_features.shape
    if len(shape) > 2:
        train_features = train_features.reshape((shape[0], shape[1] * shape[2]))

    shape = test_features.shape
    if len(shape) > 2:
        test_features = test_features.reshape((shape[0], shape[1] * shape[2]))

    print("Train feature shape : ", train_features.shape, "Classes : ", len(np.unique(y_train)))
    print("Test features shape : ", test_features.shape, "Classes : ", len(np.unique(y_test)))

    np.save(save_dir + '%s_%s_train_features.npy' % (layer_name, dataset_name), train_features)
    np.save(save_dir + '%s_%s_train_labels.npy' % (layer_name, dataset_name), y_train)
    np.save(save_dir + '%s_%s_test_features.npy' % (layer_name, dataset_name), test_features)
    np.save(save_dir + '%s_%s_test_labels.npy' % (layer_name, dataset_name), y_test)

    print("Saved train feature vectors at %s" % (save_dir + '%s_%s_train_features.npy' % (layer_name, dataset_name)))
    print("Saved test feature vectors at %s" % (save_dir + '%s_%s_test_features.npy' % (layer_name, dataset_name)))
    print()


class MaskablePermute(Permute):

    def __init__(self, dims, **kwargs):
        super(MaskablePermute, self).__init__(dims, **kwargs)
        self.supports_masking = True
