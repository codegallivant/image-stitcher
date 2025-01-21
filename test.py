import logging
import yaml
import time
from stitcher import ArbitraryStitcher, ConsecutiveStitcher
import os
import cv2
from utils.proc import load_image
import glob


# Read the configuration file
logging.info("Reading configuration file")
with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    PROJECT_DIR = cfg["project_dir"]
    INPUT_DIR = cfg["input_dir"]
    STITCH_DIR = cfg["stitch_dir"]
    MIN_MATCH_COUNT = cfg["min_match_threshold"]
    RATIO = cfg["lowes_ratio"]
    ALGORITHM = cfg["algorithm"]
    MATCHER = cfg["matcher"]
    MODE = cfg["type"]
    CONSECUTIVE_RANGE = cfg["consecutive_range"]
    INPUT_LIMIT = cfg["input_limit"]
    ARBITRARY = cfg["arbitrary"]
    REF_IMAGE_CONTRIB = cfg["ref_image_contrib"]
    if cfg['resize'] == None:
        RESIZE = None
    else:
        RESIZE = [int(x) for x in cfg["resize"].split('x')]
    LOGLEVEL = int(cfg["loglevel"])


if __name__ == "__main__":

    INPUT_DIR = "/mnt/d/datasets/isdc_test_drone_datasets/d1"
    files = os.listdir(INPUT_DIR)
    files = [os.path.join(INPUT_DIR, file) for file in files]

    files = [file for file in files if file[-4:]==".png" or file[-4:]==".jpg"]

    files = [p[1] for p in enumerate(files) if p[0]%1==0]

    def sort_key(s, a):
        p = os.path.basename(s)[:-4].split('_')
        p = [x.split('.') for x in p]
        pf = list()
        for xs in p:
            for x in xs:
                pf.append(x)
        return int(pf[a])

    filepaths = sorted(files, key=lambda x: (sort_key(x, 1), sort_key(x, 2), sort_key(x, 3)))[:INPUT_LIMIT]

    kwargs = {
        "filepaths": filepaths,
        "min_match_count": MIN_MATCH_COUNT,
        "lowes_ratio_threshold": RATIO,
        "algorithm": ALGORITHM,
        "matcher": MATCHER,
        "type": MODE,
        "resize": RESIZE, 
        "log_level": LOGLEVEL,
        "consecutive_range": CONSECUTIVE_RANGE,
        "ref_image_contrib": REF_IMAGE_CONTRIB
    }

    if REF_IMAGE_CONTRIB is None:
        del kwargs["ref_image_contrib"]

    if ARBITRARY:
        # Test arbitrary stitching
        arbs = ArbitraryStitcher(**kwargs)
        collections = arbs.get_collections()
        stitched = arbs.stitch_collections(collections)
    else:
        # Test consecutive stitching
        del kwargs["filepaths"]
        del kwargs["consecutive_range"]
        cons = ConsecutiveStitcher(**kwargs)
        start_time = time.time()
        for filepath in filepaths:
            cons.stitch(load_image(filepath))
        end_time = time.time()
        print("input image count:", len(filepaths))
        print("duration:", end_time - start_time)
        print("rate:", (end_time - start_time)/len(filepaths), "s/stitch")
        print("rate:", len(filepaths)/(end_time - start_time), "stitch/s")
        stitched = cons.refs

    delfiles = glob.glob(f'{STITCH_DIR}/*')
    for f in delfiles:
        os.remove(f)
    for i, stitch in enumerate(stitched):
        cv2.imwrite(os.path.join(STITCH_DIR, f"{i}.png"), stitch)
    cv2.destroyAllWindows()
