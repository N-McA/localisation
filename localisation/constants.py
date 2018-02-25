
from pathlib import Path

data_loc = Path('/local/sdd/nm583/extracted_kitti_data')
train_proportion = 0.8
kitti_image_shape = 128, 423
dubrovnik_image_shape = 224, 224
embedding_size = 128

# image_cache = CachingImageIterator(
#     # Hacky, cache works without paths but does not iterate.
#     paths=[],
#     cache_path='/local/sdd/nm583/caches/kitti-cache',
#     image_type='jpg',
#     storage_size=kitti_image_shape,
# )
