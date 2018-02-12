
import pickle
import numpy as np

import utils
import constants
import mobilenet
import layers

from keras.models import load_model


weight_loc = constants.data_loc / 'weights'
model_path = sorted(weight_loc.glob('*'))[-1]
print('model path', model_path)

embedding_model = load_model(
    model_path,
    custom_objects={
        'relu6': mobilenet.relu6,
        'DepthwiseConv2D': mobilenet.DepthwiseConv2D,
        'LearnScale': layers.LearnScale,
    }
)


def batchify(g, n):
    b = []
    for x in g:
        b.append(x)
        if len(b) == n:
            yield b
            b = []
    if len(b) > 0:
        yield b


def get_img(path):
    return constants.image_cache.cached_fetch(path)


def compute_embedding(paths):
    path_to_index = {p: i for i, p in enumerate(paths)}
    embeddings = np.zeros([len(paths), constants.embedding_size])
    i = 0
    shape = None
    for batch_paths in batchify(paths, 64):
        if shape is None:
            img = get_img(batch_paths[0])
            shape = img.shape
        batch = np.zeros([len(batch_paths), *shape])
        for j, path in enumerate(batch_paths):
            batch[j] = get_img(path)
        batch_embeddings = \
            embedding_model.predict(batch, batch_size=len(batch))
        embeddings[i:i+len(batch)] = batch_embeddings
        i += len(batch)
    embeddings /= np.linalg.norm(embeddings, axis=-1)[:, np.newaxis]
    return path_to_index, embeddings


def dot_product_search(
        path,
        k,
        path_to_index,
        embedded_paths,
        embeddings):

    idx = path_to_index[path]
    close_indexes = \
        np.argsort(-np.sum(embeddings[idx] * embeddings, axis=-1))[1:11]
    return [embedded_paths[i] for i in close_indexes]


def evaluate_mapk(search_method, query_paths, relation):
    gt_results = []
    pred_results = []
    for p in query_paths:
        pred_results.append(search_method(p))
        gt_results.append(list(relation[p]))

    return utils.mapk(gt_results, pred_results)


with (constants.data_loc / 'graph.pickle').open('rb') as f:
    whole_relation = pickle.load(f)


test_paths = utils.load_paths(constants.data_loc / 'test_paths.txt')
test_path_to_index, test_embeddings = compute_embedding(test_paths)
search = lambda p: dot_product_search(
        p, 10, test_path_to_index, test_paths, test_embeddings)

np.random.seed(7)
query_paths = np.random.choice(test_paths, 100)
print(evaluate_mapk(search, query_paths, whole_relation))
