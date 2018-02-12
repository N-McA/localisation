
from pathlib import Path

import sys
sys.path.append('/home/nm583/part-3-project/multimodal')
from multimodal.image_iterators import CachingImageIterator

data_loc = Path('/local/sdd/nm583/extracted_kitti_data')
train_proportion = 0.8
kitti_image_shape = 128, 423
embedding_size = 100


image_cache = CachingImageIterator(
    # Hacky, cache works without paths but does not iterate.
    paths=[],
    cache_path='/local/sdd/nm583/caches/kitti-cache',
    image_type='jpg',
    storage_size=kitti_image_shape,
)
