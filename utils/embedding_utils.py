import numpy as np
import h5py


def extract_embeddings(dataset_prefix, weights_path):
    f = h5py.File(weights_path)
    g = f['embedding_1']
    embedding_layer_name = str(g.attrs['weight_names'], encoding='utf8')
    weights = g[embedding_layer_name][:]

    print("Embedding extracted and saved")
    np.save('./embeddings/%s_vectors.npy' % (dataset_prefix), weights)


if __name__ == "__main__":
    pass



