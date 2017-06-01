import os
import numpy as np

from gensim.models import Word2Vec
from utils.generic_utils import load_dataset_at
from utils.constants import MAX_NB_WORDS_LIST

class NumpyArrayIterator:

    def __init__(self, X):
        self.X = X
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.X.shape[0]:
            val = self.X[self.count].astype('str').tolist()
            self.count += 1
            return val
        else:
           raise StopIteration()


def create_vectors(dataset_id, embedding_size, dataset_prefix):
    X_train, _, _, _ = load_dataset_at(dataset_id)

    iterator = NumpyArrayIterator(X_train)

    window_size = MAX_NB_WORDS_LIST[dataset_id] - 1 # every 'word' can depend on evey other word

    print('Training Word2Vec model')
    model = Word2Vec(iterator, size=embedding_size, min_count=2, workers=4, iter=100, window=window_size)


    print('Saving Word2Vec model')
    model.wv.save_word2vec_format('../embeddings/%s_vectors.txt' % dataset_prefix, binary=False)


def load_embeddings(dataset_prefix, verbose=True):
    txt_path = '../embeddings/%s_vectors.txt' % (dataset_prefix)
    npy_path = txt_path[:-3] + "npy"

    if os.path.exists(txt_path) or os.path.exists(npy_path):
        return __load_embeddings(dataset_prefix, txt_path, npy_path, verbose)
    elif os.path.exists(txt_path[1:]) or os.path.exists(npy_path[1:]):
        return __load_embeddings(dataset_prefix, txt_path[1:], npy_path[1:], verbose)
    else:
        raise FileNotFoundError('Word2Vec embedding file for dataset \'%s\' could not be found in the embedding directory' %
                                (dataset_prefix))


def __load_embeddings(dataset_prefix, txt_path, npy_path, verbose=False):

    embedding_path = npy_path # change to numpy data format (which contains the preloaded embedding matrix)
    if os.path.exists(embedding_path):
        # embedding matrix exists, no need to create again.
        print("Loading embedding matrix for dataset \'%s\'" % (dataset_prefix))
        embedding_matrix = np.load(embedding_path)
        return embedding_matrix

    with open(txt_path, 'r', encoding='utf8') as f:
        header = f.readline()
        splits = header.split(' ')

        vocab_size = int(splits[0])
        embedding_size = int(splits[1])

        embeddings_index = {}
        error_words = []

        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except Exception:
                error_words.append(word)

        if len(error_words) > 0:
            print("%d words were not added." % (len(error_words)))
            if verbose:
                print("Words are : \n", error_words)

        if verbose: print('Preparing embedding matrix.')

        embedding_matrix = np.zeros((vocab_size, embedding_size))

        for key, vector in embeddings_index.items():
            if vector is not None:
                # words not found in embedding index will be all-zeros.
                key = int(key)
                embedding_matrix[key] = vector

        if verbose: print('Saving embedding matrix for dataset \'%s\'' % (dataset_prefix))

        np.save(embedding_path, embedding_matrix)
        return embedding_matrix


if __name__ == "__main__":
    create_vectors(dataset_id=2, embedding_size=300, dataset_prefix='beef')

    load_embeddings(dataset_prefix='beef')



