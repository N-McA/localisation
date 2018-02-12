
import pickle
import argparse
import numpy as np

import utils
import constants
import mobilenet

from keras.models import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default=None)
    return p.parse_args()


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


def compute_embedding(
        paths, 
        embedding_model, 
        embedding_size=constants.embedding_size):
    path_to_index = {p: i for i, p in enumerate(paths)}
    embeddings = np.zeros([len(paths), embedding_size])
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


def evaluate_trained_embedding_model(args, test_paths, query_paths, whole_relation):
    weight_loc = constants.data_loc / 'weights'
    if args is None:
        model_path = sorted(weight_loc.glob('*'))[-1]
    else:
        model_path = args.weights
    print('model path', model_path)

    embedding_model = load_model(
        model_path,
        custom_objects={
            'relu6': mobilenet.relu6,
            'DepthwiseConv2D': mobilenet.DepthwiseConv2D,
        }
    )
    test_path_to_index, test_embeddings = compute_embedding(
        test_paths, embedding_model
    )

    def search(p):
        return dot_product_search(
            p, 10, test_path_to_index, test_paths, test_embeddings)

    print('trained embedding model')
    print(evaluate_mapk(search, query_paths, whole_relation))


def evaluate_raw_mobilenet(test_paths, query_paths, whole_relation):

    embedding_model = mobilenet.MobileNet(
        input_shape=[*constants.kitti_image_shape, 3],
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    test_path_to_index, test_embeddings = compute_embedding(
        test_paths, embedding_model, embedding_size=1024
    )

    def search(p):
        return dot_product_search(
            p, 10, test_path_to_index, test_paths, test_embeddings)

    print('raw mobilenet model')
    print(evaluate_mapk(search, query_paths, whole_relation))


if __name__ == '__main__':

    args = parse_args()

    with (constants.data_loc / 'graph.pickle').open('rb') as f:
        whole_relation = pickle.load(f)

    test_paths = utils.load_paths(constants.data_loc / 'train_paths.txt')
    np.random.seed(7)
    query_paths = np.random.choice(test_paths, 100)

    evaluate_raw_mobilenet(
        test_paths,
        query_paths,
        whole_relation
    )

    evaluate_trained_embedding_model(
        args,
        test_paths,
        query_paths,
        whole_relation
    )
