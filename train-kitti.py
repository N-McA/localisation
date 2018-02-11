
import sys
sys.path.append('/home/nm583/part-3-coursework/cv/triplet-reid/')
from loss import batch_hard, cdist
sys.path.append('/home/nm583/part-3-project/multimodal')
from multimodal.image_iterators import CachingImageIterator

from mobilenet import MobileNet

import numpy as np
from pathlib import Path
from skimage.io import imread
import pickle

import keras
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Add, Reshape, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.losses import binary_crossentropy

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# So we can see memory usage...
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


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


class TripletMetricIterator(CachingImageIterator):
    def __init__(self,
                 neighbour_fetcher,
                 n_anchors,
                 n_positive_examples,
                 *args, **kwargs):
        self.neighbour_fetcher = neighbour_fetcher
        self.n_anchors = n_anchors
        self.n_positive_examples = n_positive_examples
        batch_size = n_anchors * (1 + n_positive_examples)
        super().__init__(*args, batch_size=batch_size, **kwargs)

    def _batch_shape(self, index_array):
        n_rows = len(index_array)*(self.n_positive_examples + 1)
        return (n_rows, *self.storage_size, self.n_channels)

    def _get_batches_of_transformed_samples(self, index_array):
        index_array = index_array[:self.n_positive_examples]
        batch = np.zeros(self._batch_shape(index_array))
        batch_root_paths = self.paths[index_array]
        i = 0
        for root_path in batch_root_paths:
            im = self.cached_fetch(root_path)
            batch[i] = self.augment(im)
            i += 1
            for neigh_path in self.neighbour_fetcher(root_path, self.n_positive_examples):
                im = self.cached_fetch(neigh_path)
                batch[i] = self.augment(im)
                i += 1
        repped_ids = np.repeat(index_array, [self.n_positive_examples + 1])
        return [repped_ids, batch], np.zeros(len(batch))


def training_generator(n_anchors, n_positives):
    caching_image_iter = CachingImageIterator(
        paths=train_paths,
        cache_path='/local/sdd/nm583/caches/kitti-cache',
        image_type='jpg',
        storage_size=kitti_image_shape,
    )
    while True:
        batch = None
        for path in train_paths:
            if batch is None:
                batch = np.zeros([n_anchors*(n_positives + 1), *kitti_image_shape, 3])
                i = 0
            batch[i] = caching_image_iter.cached_fetch(path)
            i += 1
            for neigh_path in graph_neigh_fetcher(path, n_positives):
                batch[i] = caching_image_iter.cached_fetch(path)
                i += 1
            if i == len(batch):
                yield batch
                batch = None
        np.random.shuffle(train_paths)

n_bits = 100

mobile_net_model = MobileNet(input_shape=[*kitti_image_shape, 3], include_top=False, weights='imagenet')

hashing_net_input = Input([*kitti_image_shape, 3], dtype=np.float32)
hashing_net = hashing_net_input
hashing_net = mobile_net_model(hashing_net_input)
hashing_net = GlobalAveragePooling2D()(hashing_net)
hashing_net = Dense(
    n_bits, activation='sigmoid'
)(hashing_net)
hashing_model = Model([hashing_net_input], [hashing_net])


triplet_model_id_input = Input([1], dtype=np.int32)
triplet_model_data_input = Input([*kitti_image_shape, 3], dtype=np.float32)
hashes = hashing_model(triplet_model_data_input)

def keras_batch_hard_triplet_loss(pids_embeddings):
    pids, embeddings = pids_embeddings
    dists = cdist(embeddings, embeddings, 'cosine')
    diff= batch_hard(dists, pids, 'soft')
    return diff

loss = Lambda(keras_batch_hard_triplet_loss)([triplet_model_id_input, hashes])

triplet_model = Model([triplet_model_id_input, triplet_model_data_input], [loss])


triplet_iter = TripletMetricIterator(
    neighbour_fetcher=graph_neigh_fetcher,
    n_anchors=8,
    n_positive_examples=6,
    paths=train_paths,
    cache_path='/local/sdd/nm583/caches/kitti-cache',
    image_type='jpg',
    storage_size=kitti_image_shape,
)


def wrap_triplet_iter():
    # madness I tell ya
    for x in triplet_iter:
        yield x


opt = keras.optimizers.Adam()
triplet_model.compile(opt, lambda _, x: x)
triplet_model.fit_generator(wrap_triplet_iter(), steps_per_epoch=len(train_paths), epochs=1)

train_paths = sorted(train_paths)
all_fts = []
for path in train_paths:
    b = np.array([triplet_iter.cached_fetch(path)])
    all_fts.append(hashing_model.predict(b)[0])
all_fts = np.array(all_fts)
np.save('features.npy', all_fts)

