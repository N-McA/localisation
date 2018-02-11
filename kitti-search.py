
import sys
sys.path.append('/home/nm583/part-3-project/multimodal')
from multimodal.image_iterators import CachingImageIterator

from mobilenet import MobileNet

import numpy as np
from pathlib import Path
from skimage.io import imread
import pickle

import keras
from keras.models import Model
from keras.layers import (
        Dense, Input, Lambda,
        Add, Reshape, Dropout,
        Activation
)
from keras.layers import GlobalAveragePooling2D
from keras.losses import binary_crossentropy
from keras.engine.topology import Layer

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# So we can see memory usage...
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


class LearnScale(Layer):

    def __init__(self, initial_value, **kwargs):
        self.initial_value = initial_value
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.scale = self.add_weight(
            name='scale',
            shape=[1],
            initializer=keras.initializers.Constant(self.initial_value),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape


with open('/local/sdd/nm583/graph.pickle', 'rb') as f:
    path_graph = pickle.load(f)
    path_graph = {k: list(path_graph[k]) for k in path_graph}

train_paths = sorted(list(path_graph.keys()))

#  img = imread(train_paths[0])
#  kitti_image_shape = 224, int(224 * img.shape[1] / img.shape[0])
#  del img
kitti_image_shape = 128, 423

path_to_index = {path: i for i, path in enumerate(train_paths)}


def graph_neigh_fetcher(path, n):
    return np.random.choice(path_graph[path], replace=False, size=n)


def path_batch_generator(paths, batch_size):
    batch = []
    while True:
        for path in paths:
            batch.append(path)
            if len(batch) == batch_size:
                yield batch
                batch = []
        np.random.shuffle(paths)


def training_generator(paths, n_positives, relation):
    cache = CachingImageIterator(
        paths=train_paths,
        cache_path='/local/sdd/nm583/caches/kitti-cache',
        image_type='jpg',
        storage_size=kitti_image_shape,
    )
    for path_batch in path_batch_generator(paths, n_positives):
        positives_a = []
        positives_b = []
        for path in path_batch:
            neighs = relation[path]
            if len(neighs) > 0:
                positives_a.append(path)
                positives_b.append(np.random.choice(neighs))
        b = len(positives_a)
        images_a = np.zeros([b, *kitti_image_shape, 3])
        images_b = np.zeros([b, *kitti_image_shape, 3])
        for i, (pa, pb) in enumerate(zip(positives_a, positives_b)):
            images_a[i] = cache.augment(cache.cached_fetch(pa))
            images_b[i] = cache.augment(cache.cached_fetch(pb))

        m = np.zeros([len(positives_a), len(positives_b)])
        for i, path_a in enumerate(positives_a):
            for j, path_b in enumerate(positives_b):
                if path_b in relation[path_a]:
                    m[i, j] = 1.0

        yield [images_a, images_b], m


n_bits = 100

mobile_net_model = MobileNet(
    input_shape=[*kitti_image_shape, 3],
    include_top=False,
    weights='imagenet'
)

embedding_net_input = Input([*kitti_image_shape, 3], dtype=np.float32)
embedding_net = embedding_net_input
embedding_net = mobile_net_model(embedding_net_input)
embedding_net = GlobalAveragePooling2D()(embedding_net)
embedding_net = Dense(
    n_bits, activation='linear'
)(embedding_net)
embedding_model = Model([embedding_net_input], [embedding_net])


input_a = Input([*kitti_image_shape, 3])
input_b = Input([*kitti_image_shape, 3])

input_a_embedded = embedding_model(input_a)
input_b_embedded = embedding_model(input_b)


def all_dot_prods(a, b):
    return tf.reduce_sum(tf.expand_dims(a, axis=1) * tf.expand_dims(b, axis=0), axis=-1)


pairwise_dot_prods = Lambda(lambda ab: all_dot_prods(*ab))([input_a_embedded, input_b_embedded])
pairwise_dot_prods = LearnScale(0.01)(pairwise_dot_prods)
sigma_pairwise_dot_prods = Activation('sigmoid')(pairwise_dot_prods)

pairwise_model = Model(
    [input_a, input_b],
    [sigma_pairwise_dot_prods]
)

# opt = keras.optimizers.Adam(amsgrad=True)
opt = keras.optimizers.RMSprop()
pairwise_model.compile(opt, loss='binary_crossentropy')

relation = path_graph
tg = training_generator(
        paths=list(relation.keys()),
        n_positives=6,
        relation=relation,
)

pairwise_model.fit_generator(tg, steps_per_epoch=len(relation), epochs=1)
#  for x, y in tg:
#      print(pairwise_model.predict(x))
#      break

#  train_paths = sorted(train_paths)
#  all_fts = []
#  for path in train_paths:
#      b = np.array([triplet_iter.cached_fetch(path)])
#      all_fts.append(embedding_model.predict(b)[0])
#  all_fts = np.array(all_fts)
#  np.save('features.npy', all_fts)

