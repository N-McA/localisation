
import sys
sys.path.append('/home/nm583/part-3-project/multimodal')
from multimodal.image_iterators import CachingImageIterator

image_cache = CachingImageIterator(
    # Hacky, cache works without paths but does not iterate.
    paths=[],
    cache_path='/local/sdd/nm583/caches/dubrovnik-cache',
    image_type='jpg',
    storage_size=[224],
)

