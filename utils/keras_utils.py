import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-paper')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

from keras.models import Model
from keras.layers import Permute
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

from utils.generic_utils import load_dataset_at, calculate_dataset_metrics, cutoff_choice, \
                                cutoff_sequence, plot_dataset
from utils.constants import MAX_SEQUENCE_LENGTH_LIST


def train_model(model:Model, dataset_id, dataset_prefix, epochs=50, batch_size=128, val_subset=None,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-3):
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

    model_checkpoint = ModelCheckpoint("./weights/%s_weights.h5" % dataset_prefix, verbose=1,
                                       monitor='val_acc', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=100, mode='max',
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))


def evaluate_model(model:Model, dataset_id, dataset_prefix, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
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

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    print("\nEvaluating : ")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
    print()
    print("Final Accuracy : ", accuracy)

    return accuracy


def hyperparameter_search_over_model(model_gen, dataset_id, param_grid, cutoff=None,
                                     normalize_timeseries=False):

    X_train, y_train, _, _, is_timeseries = load_dataset_at(dataset_id,
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
            X_train, _ = cutoff_sequence(X_train, None, choice, dataset_id, sequence_length)

    if not is_timeseries:
        print("Model hyper parameters can only be searched for time series models")
        return

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                                 np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    y_train = to_categorical(y_train, len(np.unique(y_train)))

    clf = KerasClassifier(build_fn=model_gen,
                          epochs=50,
                          class_weight=class_weight,
                          verbose=0)

    grid = GridSearchCV(clf, param_grid=param_grid,
                        n_jobs=1, verbose=10, cv=3)

    result = grid.fit(X_train, y_train)

    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def set_trainable(layer, value):
   layer.trainable = value

   # case: container
   if hasattr(layer, 'layers'):
       for l in layer.layers:
           set_trainable(l, value)

   # case: wrapper (which is a case not covered by the PR)
   if hasattr(layer, 'layer'):
        set_trainable(layer.layer, value)


def build_function(model, layer_names=None, outputs=None):
    inp = model.input

    if layer_names is not None and (type(layer_names) != list and type(layer_names) != tuple):
        layer_names = [layer_names]

    if outputs is None:
        if layer_names is None:
            outputs = [layer.output for layer in model.layers] # all layer outputs
        else:
            outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    else:
        outputs = outputs

    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    return funcs


def get_outputs(model, inputs, eval_functions, verbose=False):
    if verbose: print('----- activations -----')
    outputs = []
    layer_outputs = [func([inputs, 1.])[0] for func in eval_functions]
    for layer_activations in layer_outputs:
        outputs.append(layer_activations)
    return outputs


def visualise_attention(model:Model, dataset_id, dataset_prefix, layer_name, cutoff=None, limit=None,
                        normalize_timeseries=False, visualize_sequence=True, visualize_classwise=False):
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
            X_train , X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    eval_functions = build_function(model, layer_name)
    attention_vectors = []

    for i in range(X_train.shape[0]):
        activations = get_outputs(model,
                                  X_train[i, :, :][np.newaxis, ...],
                                  eval_functions,
                                  verbose=False)[0]

        if activations.shape[-1] != 1: # temporal attention
            axis = 1
        else:
            axis = -1 # spatial attention

        attention_vector = np.mean(activations, axis=axis).squeeze()

        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vectors = np.array(attention_vectors)
    attention_vector_final = np.mean(attention_vectors, axis=0)

    if visualize_sequence:
        # plot input sequence part that is paid attention too in detail
        attention_vector_final = attention_vector_final.reshape((1, attention_vector_final.shape[0]))

        X_train_attention = np.zeros_like(X_train)
        X_test_attention = np.zeros_like(X_test)

        for i in range(X_train.shape[0]):
            X_train_attention[i, :, :] = attention_vector_final * X_train[i, :, :]

        for i in range(X_test.shape[0]):
            X_test_attention[i, :, :] = attention_vector_final * X_test[i, :, :]

        plot_dataset(dataset_id, seed=1, limit=limit, cutoff=cutoff,
                     normalize_timeseries=normalize_timeseries, plot_data=(X_train, y_train, X_test, y_test,
                                                                           X_train_attention, X_test_attention),
                     type='Attention', plot_classwise=visualize_classwise)

    else:
        # plot only attention chart

        train_df = pd.DataFrame({'attention (%)': attention_vector_final},
                          index=range(attention_vector_final.shape[0]))

        train_df.plot(kind='bar',
                title='Attention Mechanism (Train) as '
                'a function of input'
                ' dimensions.')

        plt.show()


def visualize_context_vector(model:Model, dataset_id, dataset_prefix, cutoff=None, limit=None,
                             normalize_timeseries=False, visualize_sequence=True, visualize_classwise=False):
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
            X_train , X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    attn_lstm_layer = [(i, layer) for (i, layer) in enumerate(model.layers)
                       if layer.__class__.__name__ == 'AttentionLSTM']

    if len(attn_lstm_layer) == 0:
        raise AttributeError('Provided model does not have an Attention layer')
    else:
        i, attn_lstm_layer = attn_lstm_layer[0] # use first attention lstm layer only

    attn_lstm_layer.return_attention = True

    model.layers[i] = attn_lstm_layer
    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    attention_output = model.layers[i].call(model.input)

    eval_functions = build_function(model, attn_lstm_layer.name, outputs=[attention_output])
    attention_vectors = []

    for i in range(X_train.shape[0]):
        activations = get_outputs(model,
                                  X_train[i, :, :][np.newaxis, ...],
                                  eval_functions,
                                  verbose=False)[0]

        if activations.shape[-1] != 1: # temporal attention
            axis = 1
        else:
            axis = -1 # spatial attention

        attention_vector = np.mean(activations, axis=axis).squeeze()
        attention_vectors.append(attention_vector)

    attention_vectors = np.array(attention_vectors)
    attention_vector_final = np.mean(attention_vectors, axis=0)

    attention_vector_final = (attention_vector_final - attention_vector_final.mean()) / (attention_vector_final.std())

    if visualize_sequence:
        # plot input sequence part that is paid attention too in detail
        attention_vector_final = attention_vector_final.reshape((1, attention_vector_final.shape[0]))

        X_train_attention = np.zeros_like(X_train)
        X_test_attention = np.zeros_like(X_test)

        for i in range(X_train.shape[0]):
            X_train_attention[i, :, :] = attention_vector_final * X_train[i, :, :]

        for i in range(X_test.shape[0]):
            X_test_attention[i, :, :] = attention_vector_final * X_test[i, :, :]

        plot_dataset(dataset_id, seed=1, limit=limit, cutoff=cutoff,
                     normalize_timeseries=normalize_timeseries, plot_data=(X_train, y_train, X_test, y_test,
                                                                           X_train_attention, X_test_attention),
                     type='Context', plot_classwise=visualize_classwise)

    else:
        # plot only attention chart

        train_df = pd.DataFrame({'attention (%)': attention_vector_final},
                          index=range(attention_vector_final.shape[0]))

        train_df.plot(kind='bar',
                title='Attention Mechanism (Train) as '
                'a function of input'
                ' dimensions.')

        plt.show()



def visualize_cam(model:Model, dataset_id, dataset_prefix, class_id,
                  cutoff=None, normalize_timeseries=False):
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

    conv_out = conv_out[0, :, :]
    conv_out = conv_out.transpose((1, 0))
    conv_channels = conv_out.shape[0]

    conv_cam = np.zeros(sequence_length, dtype=np.float32)
    for i, w in enumerate(class_weights[conv_channels:, class_id]):
        conv_cam += w * conv_out[i, :]

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

class MaskablePermute(Permute):

    def __init__(self, dims, **kwargs):
        super(MaskablePermute, self).__init__(dims, **kwargs)
        self.supports_masking = True


