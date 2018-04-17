
import utils
from callbacks import ModelCheckpoint, DubrovnikEvalCallback, CSVLoggingCallback
from mobilenet import MobileNet
import constants
from constants import data_loc, train_proportion, embedding_size

from layers import LearnScale

import cornell_visibility_graph as cvg
from image_cache import image_cache

import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict

import keras
from keras.models import Model
from keras.layers import (
        Dense, Input, Lambda,
        Activation, BatchNormalization, LeakyReLU
)
from keras.layers import GlobalAveragePooling2D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

image_shape = 224, 224
TRAIN_CITY = 'DUBROVNIK'

if TRAIN_CITY == 'DUBROVNIK':
    geo_locs = cvg.get_dubrovnik_geo_locs()
    whole_relation = cvg.get_dubrovnik_relation()
if TRAIN_CITY == 'ROME':
    geo_locs = cvg.get_rome_geo_locs()
    whole_relation = cvg.get_rome_relation()
    # There's one pesky bad image...
    bad = Path('/home/nm583/sfm_data/Rome16K/db/7504080@N05_2397193450.jpg')
    del whole_relation[bad]
    for neighs in whole_relation.values():
        if bad in neighs:
            neighs.remove(bad)


# So we can see memory usage...
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

all_train_paths = np.array(list(whole_relation.keys()))

np.random.shuffle(all_train_paths)
n_train_paths = int(train_proportion * len(all_train_paths))
train_paths = all_train_paths[:n_train_paths]
set_train_paths = set(train_paths)
val_paths = all_train_paths[n_train_paths:]
train_relation = {}
for path in train_paths:
    train_relation[path] = \
        set(n for n in whole_relation[path] if n in set_train_paths)

close_relation = defaultdict(set)     
for p1, neighs in whole_relation.items():
    for p2 in neighs:
        if np.linalg.norm(geo_locs[p1] - geo_locs[p2]) < 5:
            close_relation[p1].add(p2)  
    
close_train_relation = defaultdict(set)     
for p1, neighs in train_relation.items():
    for p2 in neighs:
        if np.linalg.norm(geo_locs[p1] - geo_locs[p2]) < 5:
            close_train_relation[p1].add(p2) 

print('close train relation length', len(close_train_relation))


def path_batch_generator(paths, batch_size):
    batch = []
    while True:
        for path in paths:
            batch.append(path)
            if len(batch) == batch_size:
                yield batch
                batch = []
        np.random.shuffle(paths)


def get_img(path):
    return image_cache.cached_fetch(path)


def batch_generator(paths, n_locs, n_per_loc, sample_relation, relation):
    # not satisfactory...
    choosable_relation = {k: list(ns - set([k])) for k, ns in sample_relation.items()}
    for path_batch in path_batch_generator(paths, n_locs):
        full_batch = []
        for path in path_batch:
            if path in choosable_relation:
                neighs = choosable_relation[path]
                if len(neighs) > 0:
                    full_batch.append(path)
                    n = min(n_per_loc, len(neighs))
                    full_batch += list(np.random.choice(neighs, size=n))
        images = np.zeros([len(full_batch), *image_shape, 3])
        for i, p in enumerate(full_batch):
            try:
                images[i] = get_img(p)
            except Exception as e:
                print(p)
                print(e)

        m = np.zeros([len(full_batch), len(full_batch)], dtype=bool)
        for i, path_a in enumerate(full_batch):
            for j, path_b in enumerate(full_batch):
                if path_b in relation[path_a] or path_a == path_b:
                    m[i, j] = True

        yield images, m


mobile_net_model = MobileNet(
    input_shape=[*image_shape, 3],
    include_top=False,
    weights='imagenet',
    pooling='avg',
    alpha=0.5,
)

# for layer in mobile_net_model.layers:
    # layer.trainable = False

embedding_net_input = Input([*image_shape, 3], dtype=np.float32)
embedding_net = embedding_net_input
embedding_net = mobile_net_model(embedding_net_input)
embedding_net = BatchNormalization()(embedding_net)
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
        margin = 0.05
        return tf.nn.softplus(diff + margin)


pair_net_input = Input([*image_shape, 3])
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

#  embedding_dense_layer.set_weights([np.eye(1024), np.zeros(1024)])

# # Pre-processing to compute init.
# preproc_paths = np.random.choice(train_paths, len(train_paths))
# imgs = np.array([get_img(p) for p in train_paths])
# embeds = mobile_net_model.predict(imgs)
# from sklearn.decomposition import PCA
# pca = PCA(embedding_size)
# pca_init = pca.fit(embeds)
# embedding_dense_layer.set_weights([pca.components_.T, np.zeros(embedding_size)])

n_close = 24
tg = batch_generator(
        paths=train_paths,
        n_locs=n_close,
        n_per_loc=3,
        relation=close_train_relation,
        sample_relation=close_train_relation,
)

vg = batch_generator(
    paths=val_paths,
    n_locs=6,
    n_per_loc=3,
    relation=close_relation,
    sample_relation=close_relation
)

if __name__ == '__main__':
    embedding_model.summary()

    pairwise_model.fit_generator(
            tg,
            steps_per_epoch=len(train_relation)//n_close,
            # steps_per_epoch=100,
            epochs=100,
            validation_data=vg,
            validation_steps=100,
            callbacks=[
                ModelCheckpoint('/local/scratch/nm583/sfm_data/weights', embedding_model),
                DubrovnikEvalCallback(embedding_model),
                CSVLoggingCallback(Path('/local/scratch/nm583/sfm_data/log.ndjson').expanduser())
            ]
    )
