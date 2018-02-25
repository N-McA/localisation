

import utils
from callbacks import ModelCheckpoint
from mobilenet import MobileNet
import constants
from constants import data_loc, train_proportion, kitti_image_shape, embedding_size

from layers import LearnScale

import numpy as np
from pathlib import Path
import pickle

import keras
from keras.models import Model
from keras.layers import (
        Dense, Input, Lambda,
        Activation, BatchNormalization, LeakyReLU
)
from keras.layers import GlobalAveragePooling2D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# So we can see memory usage...
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


with (data_loc / 'graph.pickle').open('rb') as f:
    path_graph = pickle.load(f)

with (data_loc / 'loop_closure_graph.pickle').open('rb') as f:
    loop_closure_graph = pickle.load(f)
all_train_paths = utils.load_paths(data_loc / 'train_paths.txt')
test_paths = utils.load_paths(data_loc / 'test_paths.txt')

np.random.shuffle(all_train_paths)
n_train_paths = int(train_proportion * len(all_train_paths))
train_paths = all_train_paths[:n_train_paths]
set_train_paths = set(train_paths)
val_paths = all_train_paths[n_train_paths:]
whole_relation = path_graph
train_relation = {}
for path in train_paths:
    train_relation[path] = \
        set(n for n in whole_relation[path] if n in set_train_paths)


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
    return constants.image_cache.cached_fetch(path)


def batch_generator(paths, n_positives, relation):
    # not satisfactory...
    choosable_relation = {k: list(ns) for k, ns in relation.items()}
    for path_batch in path_batch_generator(paths, n_positives):
        positives_a = []
        positives_b = []
        for path in path_batch:
            neighs = choosable_relation[path]
            if len(neighs) > 0:
                positives_a.append(path)
                positives_b.append(np.random.choice(neighs))
        b = len(positives_a)
        images_a = np.zeros([b, *kitti_image_shape, 3])
        images_b = np.zeros([b, *kitti_image_shape, 3])
        for i, (pa, pb) in enumerate(zip(positives_a, positives_b)):
            images_a[i] = get_img(pa)
            images_b[i] = get_img(pb)

        m = np.zeros([len(positives_a), len(positives_b)])
        for i, path_a in enumerate(positives_a):
            for j, path_b in enumerate(positives_b):
                if path_b in relation[path_a]:
                    m[i, j] = 1.0

        yield [images_a, images_b], m


mobile_net_model = MobileNet(
    input_shape=[*kitti_image_shape, 3],
    include_top=False,
    weights='imagenet',
    pooling='avg',
)

for layer in mobile_net_model.layers:
    layer.trainable = False

embedding_net_input = Input([*kitti_image_shape, 3], dtype=np.float32)
embedding_net = embedding_net_input
embedding_net = mobile_net_model(embedding_net_input)
embedding_net = BatchNormalization(scale=False)(embedding_net)
embedding_net = Dense(
    embedding_size*2, activation='linear'
)(embedding_net)
embedding_net = LeakyReLU(0.1)(embedding_net)
embedding_net = Dense(
    embedding_size, activation='linear',
)(embedding_net)
embedding_model = Model([embedding_net_input], [embedding_net])

input_a = Input([*kitti_image_shape, 3])
input_b = Input([*kitti_image_shape, 3])

input_a_embedded = embedding_model(input_a)
input_b_embedded = embedding_model(input_b)


# Pre-processing to compute init.
#  preproc_paths = np.random.choice(train_paths, len(train_paths))
#  imgs = np.array([get_img(p) for p in train_paths])
#  embeds = mobile_net_model.predict(imgs)
#  from sklearn.decomposition import PCA
#  pca = PCA(embedding_size)
#  pca_init = pca.fit(embeds)
#  embedding_dense_layer.set_weights([pca.components_.T, np.zeros(embedding_size)])



def all_prods(a, b):
    return tf.expand_dims(a, axis=1) * tf.expand_dims(b, axis=0)


def all_cosines(a, b):
    all_dot_prods = tf.reduce_sum(all_prods(a, b), axis=-1)
    #all_norm_prods = all_prods(tf.norm(a, axis=-1), tf.norm(b, axis=-1))
    return all_dot_prods #/ all_norm_prods


pairwise_cosines = Lambda(lambda ab: all_cosines(*ab))([
    input_a_embedded, input_b_embedded
])
scale_layer = LearnScale(0.6)
pairwise_cosines = scale_layer(pairwise_cosines)
sigma_pairwise_cosines = Activation('sigmoid')(pairwise_cosines)

pairwise_model = Model(
    [input_a, input_b],
    [sigma_pairwise_cosines]
)

# opt = keras.optimizers.Adam(amsgrad=True)
opt = keras.optimizers.RMSprop()
pairwise_model.compile(opt, loss='binary_crossentropy')

#  embedding_dense_layer.set_weights([np.eye(1024), np.zeros(1024)])

relation = path_graph
tg = batch_generator(
        paths=train_paths,
        n_positives=16,
        relation=train_relation,
)

vg = batch_generator(
    paths=val_paths,
    n_positives=8,
    relation=whole_relation,
)


def print_scale(a, b):
    print(scale_layer.get_weights())


pairwise_model.fit_generator(
        tg,
        steps_per_epoch=len(train_relation),
        epochs=10,
        validation_data=vg,
        validation_steps=100,
        callbacks=[
            ModelCheckpoint(data_loc / 'weights', embedding_model),
        ]
)
