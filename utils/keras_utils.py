import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from keras.models import Model
from keras.optimizers import Adam, Adadelta
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from utils.generic_utils import load_dataset_at


def train_model(model:Model, dataset_id, dataset_prefix, epochs=50, batch_size=128, test_data_subset=None):
    logs_dir = "./logs/%s/" % (dataset_prefix)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    X_train, y_train, X_test, y_test = load_dataset_at(dataset_id)

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                           np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    model_checkpoint = ModelCheckpoint("./weights/%s_weights.h5" % dataset_prefix, verbose=1,
                                       monitor='val_acc', save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(logs_dir, embeddings_freq=5, histogram_freq=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max',
                                  factor=0.8, cooldown=5, min_lr=1e-6, verbose=2)
    callback_list = [model_checkpoint, reduce_lr, tensorboard]

    optm = Adadelta()
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if test_data_subset is not None:
        X_test = X_test[:test_data_subset]
        y_test = y_test[:test_data_subset]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=1, validation_data=(X_test, y_test))

    model.load_weights("./weights/%s_weights.h5" % dataset_prefix)

    print("\nEvaluating : ")
    scores = model.evaluate(X_test, y_test, batch_size=batch_size)
    print()
    print("Final Scores : ", scores)


