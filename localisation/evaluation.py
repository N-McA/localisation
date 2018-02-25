
import pickle
import argparse
from collections import defaultdict
import json

import numpy as np

import utils
import constants
import mobilenet
import cornell_visibility_graph as cvg
import image_cache

from keras.models import load_model
from sklearn.neighbors import NearestNeighbors


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
    return image_cache.image_cache.cached_fetch(path)


def compute_embedding(
        paths, 
        embedding_model, 
        embedding_size=constants.embedding_size):
    embeddings = np.zeros([len(paths), embedding_size])
    i = 0
    shape = None
    print()
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
        print(i, len(paths), flush=True, end="\r")
    embeddings /= np.linalg.norm(embeddings, axis=-1)[:, np.newaxis]
    return embeddings


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
        input_shape=[*constants.dubrovnik_image_shape, 3],
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


def median_min_d_at_k(*, query_to_results, geo_locs, excluded_paths):
    min_d_at_k = defaultdict(list)
    for query_path, result_paths in query_to_results.items():
        if query_path in excluded_paths:
            continue
        query_loc = geo_locs[query_path]
        result_locs = np.array([geo_locs[p] for p in result_paths])
        result_ds = np.linalg.norm(query_loc - result_locs, axis=-1)
        for k in range(1, len(result_ds)):
            min_d_at_k[k].append(np.min(result_ds[:k]))
    return {k:np.median(min_d_at_k[k]) for k in min_d_at_k}


def evaluate_model_dubrovnik(model):
    def paths_to_embedding_f(paths):
        return compute_embedding(
            paths, model, embedding_size=model.outputs[0].shape[-1])
    return evaluate_pte_dubrovnik(paths_to_embedding_f)


def evaluate_results_dict_dubrovnik(results):
    image_geo_locs = cvg.get_dubrovnik_geo_locs()
    return median_min_d_at_k(
        query_to_results=results, 
        geo_locs=image_geo_locs,
        # TODO
        excluded_paths=set()
    )

def evaluate_pte_dubrovnik(paths_to_embedding_f):
    query_paths = cvg.get_dubrovnik_query_paths()
    db_paths = cvg.get_dubrovnik_db_paths()
    db_features = paths_to_embedding_f(db_paths)
    query_features = paths_to_embedding_f(query_paths)
    db_features /= np.linalg.norm(db_features, axis=-1, keepdims=True)
    query_features /= np.linalg.norm(query_features, axis=-1, keepdims=True)
    knn = NearestNeighbors(n_neighbors=100)
    knn.fit(db_features)
    top_k_idxs = knn.kneighbors(query_features, return_distance=False)
    query_to_results = {}
    for q, idxs in zip(query_paths, top_k_idxs):
        query_to_results[q] = [db_paths[idx] for idx in idxs]   

    return evaluate_results_dict_dubrovnik(query_to_results)


def evaluate_untuned_mobilenet():
    # mean (79.399177850460461, 28.784418906934775)
    # max (87.270394657782958, 30.477573915569153)

    embedding_model = mobilenet.MobileNet(
        input_shape=[*constants.dubrovnik_image_shape, 3],
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    print('JSON', json.dumps(['mean', evaluate_model_dubrovnik(embedding_model)]))
    embedding_model = mobilenet.MobileNet(
        input_shape=[*constants.dubrovnik_image_shape, 3],
        include_top=False,
        weights='imagenet',
        pooling='max'
    )
    print('JSON', json.dumps(['max', evaluate_model_dubrovnik(embedding_model)]))


if __name__ == '__main__':

    args = parse_args()
    evaluate_untuned_mobilenet()

    exit()

    # test_paths = utils.load_paths(constants.data_loc / 'test_paths.txt')
    # from cornell_visibility_graph import get_dubrovnik_relation
    # relation = get_dubrovnik_relation()
    # whole_relation = relation
    # db_paths = list(relation.keys())
    # np.random.seed(7)
    # query_paths = np.random.choice(db_paths, 100)

    # evaluate_raw_mobilenet(
    #     db_paths,
    #     query_paths,
    #     whole_relation
    # )

    # evaluate_trained_embedding_model(
    #     args,
    #     test_paths,
    #     query_paths,
    #     whole_relation
    # )
