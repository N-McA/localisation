
import numpy as np
from collections import defaultdict
from pathlib import Path

import cornell_visibility_graph as cvg
import callbacks as cbs

import keras
from keras.models import Model
from keras.layers import (
        Dense, Input, Lambda,
        Activation, BatchNormalization, LeakyReLU
)
import tensorflow as tf
from layers import LearnScale

from evaluation import evaluate_pte_dubrovnik
from constants import data_loc


train_proportion = 0.9


path_to_features = {}
db_features = np.load('/home/nm583/sfm_data/dubrovnik_mobilenet_db_features.npy')
db_paths = cvg.get_dubrovnik_db_paths()
for path, feat in zip(db_paths, db_features):
    path_to_features[path] = feat

query_features = np.load('/home/nm583/sfm_data/dubrovnik_mobilenet_query_features.npy')
query_paths = cvg.get_dubrovnik_query_paths()
for path, feat in zip(query_paths, query_features):
    path_to_features[path] = feat

geo_locs = cvg.get_dubrovnik_geo_locs()

whole_relation = cvg.get_dubrovnik_relation()
all_train_paths = np.array(list(whole_relation.keys()))

np.random.shuffle(all_train_paths)
n_train_paths = int(train_proportion * len(all_train_paths))
train_paths = all_train_paths[:n_train_paths]
set_train_paths = set(train_paths)
val_paths = all_train_paths[n_train_paths:]
train_relation = {}
for path in train_paths:
    train_relation[path] = set(n for n in whole_relation[path] if n in set_train_paths)
    
close_relation = defaultdict(set)     
for p1, neighs in whole_relation.items():
    for p2 in neighs:
        if np.linalg.norm(geo_locs[p1] - geo_locs[p2]) < 12:
            close_relation[p1].add(p2)  
    
close_train_relation = defaultdict(set)     
for p1, neighs in train_relation.items():
    for p2 in neighs:
        if np.linalg.norm(geo_locs[p1] - geo_locs[p2]) < 12:
            close_train_relation[p1].add(p2)  
                          

def path_batch_generator(paths, batch_size):
    batch = []
    while True:
        for path in paths:
            batch.append(path)
            if len(batch) == batch_size:
                yield batch
                batch = []
        np.random.shuffle(paths)


def batch_generator(paths, n_locs, n_per_loc, sample_relation, relation):
    # not satisfactory...
    choosable_relation = {k: list(ns) for k, ns in sample_relation.items()}
    for path_batch in path_batch_generator(paths, n_locs):
        full_batch = []
        for path in path_batch:
            neighs = choosable_relation[path]
            if len(neighs) > 0:
                full_batch.append(path)
                n = min(n_per_loc, len(neighs))
                full_batch += list(np.random.choice(neighs, size=n))
        features = np.zeros([len(full_batch), 1024])
        for i, p in enumerate(full_batch):
            try:
                features[i] = path_to_features[p]
            except ValueError as e:
                print(p)
                print(e)

        m = np.zeros([len(full_batch), len(full_batch)], dtype=bool)
        for i, path_a in enumerate(full_batch):
            for j, path_b in enumerate(full_batch):
                if path_b in relation[path_a] or path_a == path_b:
                    m[i, j] = True

        yield features, m


embedding_size = 100

embedding_net_input = Input([1024], dtype=np.float32)
embedding_net = embedding_net_input
embedding_net = BatchNormalization()(embedding_net)
embedding_net = Dense(
    embedding_size*2, activation='linear'
)(embedding_net)
embedding_net = LeakyReLU(0.1)(embedding_net)
embedding_net = Dense(
    embedding_size*2, activation='linear'
)(embedding_net)
embedding_net = LeakyReLU(0.1)(embedding_net)
embedding_net = Dense(
    embedding_size, activation='linear',
)(embedding_net)
embedding_model = Model([embedding_net_input], [embedding_net])


def all_differences(a, b):
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def all_distances(a, b):
    a = a / tf.norm(a, axis=-1, keep_dims=True)
    b = b / tf.norm(b, axis=-1, keep_dims=True)
    # norm(0) = nan
    ds = tf.norm(all_differences(a, b) + 1e-12, axis=-1)
    return ds


def batch_hard(positive_mask, dists):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.
    """
    with tf.name_scope("batch_hard"):
        furthest_positive = \
            tf.reduce_max(dists*tf.cast(positive_mask, tf.float32), axis=-1)
        # hacky:
        positives_large = dists + 1e15*tf.cast(positive_mask, tf.float32)
        closest_negative = \
            tf.reduce_min(positives_large, axis=-1)

        diff = furthest_positive - closest_negative
        # margin = 0.5
        # return tf.maximum(diff + margin, 0.0)        
        return tf.nn.softplus(diff)


def sigmoid(x):
    return 1.0 / (1 + tf.exp(-x))


def soft_margin(x):
    return sigmoid(5-10*x)

pair_net_input = Input([1024])
pair_net = embedding_model(pair_net_input)
pairwise_ds = Lambda(lambda ab: all_distances(*ab))([
    pair_net, pair_net
])

pairwise_model = Model(
    pair_net_input,
    pairwise_ds,
)

# opt = keras.optimizers.Adam(amsgrad=True)
opt = keras.optimizers.RMSprop()
pairwise_model.compile(opt, loss=batch_hard)

n_close = 24
tg = batch_generator(
        paths=train_paths,
        n_locs=n_close,
        n_per_loc=6,
        relation=close_train_relation,
        sample_relation=close_train_relation,
)

vg = batch_generator(
    paths=val_paths,
    n_locs=4,
    n_per_loc=4,
    relation=close_relation,
    sample_relation=close_relation
)

def paths_to_embedding(paths):
    features = np.array([path_to_features[p] for p in paths])
    return embedding_model.predict(features)

# b, _ = next(tg)
# print(pairwise_model.predict(b))
# exit()

pairwise_model.fit_generator(
        tg,
        steps_per_epoch=len(train_relation)//n_close,
        epochs=100,
        validation_data=vg,
        validation_steps=100,
        callbacks=[
            cbs.ModelCheckpoint('/home/nm583/sfm_data/top_weights/', embedding_model),
            cbs.DubrovnikEvalCallback(embedding_model, pte=paths_to_embedding),
            cbs.CSVLoggingCallback(Path('~/sfm_data/top_log.ndjson').expanduser())
        ]
)

# np.save('activations.py', pairwise_model.predict(next(tg)))