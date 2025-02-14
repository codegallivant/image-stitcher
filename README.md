# image-stitcher

Python library for general mass image stitching.

[![PyPI Latest Release](https://img.shields.io/pypi/v/image-stitcher.svg)](https://pypi.org/project/image-stitcher/)

## Setup
```bash
pip install image-stitcher
```

## Usage
An example can be found in ``tests/``.

### Arbitrary stitching
Relations in an arbitrarily arranged set of images are automatically determined. Each related collection found is stitched together.
```python
import os
from image_stitcher.stitcher import ArbitraryStitcher

dir_path = "path/to/dir"
filepaths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

arbs = ArbitraryStitcher(
    log_level = 2,
    output_dir = "stitches/",
    min_match_count = 5,
    lowes_ratio_threshold = 0.65,
    algorithm = 0,
    matcher = 0,
    type = 0,
    resize = None,
    consecutive_range = None,
    blend_processor = lambda x: x,
    blender = 1,
    ref_image_contrib = 0.2,
    images = None, 
    filepaths = dir_path, 
    with_tqdm = True
)

collections = arbs.get_collections()
stitched = arbs.stitch_collections(collections)
```

### Consecutive stitching
- Images are sequentially stitched without pre-determining relations in the large set. 
- Optimizations coded in allow this to stitch HD images at 5Hz and SD images at up to 10Hz. Suitable for applications such as live aerial drone mapping.
```python
import os
from image_stitcher.stitcher import ConsecutiveStitcher


dir_path = "path/to/dir"
filepaths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


def load_image(filepath, to_resize = None):
    image = cv2.imread(filepath)
    return resize_image(image, to_resize)

cons = ConsecutiveStitcher(
    log_level = 2,
    output_dir = "stitches/",
    min_match_count = 5,
    lowes_ratio_threshold = 0.65,
    algorithm = 0,
    matcher = 0,
    type = 0,
    resize = None,
    consecutive_range = None,
    blend_processor = lambda x: x,
    blender = 1,
    ref_image_contrib = 0.2,
    consecutive_range = 1,
    backup_interval = False,
    consecutive_volatility_threshold = 4,
)

for filepath in filepaths:
    cons.stitch_consecutive(load_image(filepath))
```

### Synchronous stitching
Consecutively stitches a second set of images using the same transformations calculated when stitching the first set. Can be used to stitch depth images from corresponding RGB images.
```python
from image_stitcher.stitcher import ConsecutiveStitcher


class SynchronizedStitcher:
    def __init__(self, rgb_args=None, depth_args=None, points=None):
        self.rgb_stitcher = ConsecutiveStitcher(**rgb_args)
        self.depth_stitcher = ConsecutiveStitcher(**depth_args)

    def stitch(self, rgb_image, depth_image = None):
        first_stitch = self.rgb_stitcher.stitch_count == 0
        tf = self.rgb_stitcher.stitch_consecutive(rgb_image)
        if not (depth_image is None):
            if not first_stitch:
                if self.rgb_stitcher.stitch_count > self.depth_stitcher.stitch_count:
                    self.depth_stitcher.save_and_reset()
            self.depth_stitcher.stitch_consecutive(depth_image, tf)
        return tf


sync_st = SynchronizedStitcher(
    rgb_args={
        "output_dir": "path/to/rgbdir",
        "matcher": 1,
        "algorithm": 1,
        "backup_interval": 50,
        "consecutive_range": 2,
        "blender": 1,
        "log_level": 2
    },
    depth_args={
        "output_dir": "path/to/depthdir",
        # "blend_processor": lambda x: utils.color_map(x),
        "backup_interval": 50,
        "blender": 0,
        "ref_image_contrib": 0.5,
        "log_level": 2
    }
)
```



