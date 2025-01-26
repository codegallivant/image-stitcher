import os
import sys
import numpy as np
import cv2
import copy
import glob
import time
import gc
import logging
from scipy.ndimage import distance_transform_edt

from .utils.logger import LogWrapper
from .utils.structs import DynamicConnectivity, LazyList, ConstantLengthList
from .utils.proc import load_image, get_corners_from_image, transform_keypoints_from_roi, get_padded_images, seamless_merge, crop_image, selective_color_blur, get_roi_from_corners, resize_image, seamless_merge_into_roi, seamless_gradient_merge
from .utils.utils import tqdm, no_tqdm

# from utils.logger import LogWrapper
# from utils.structs import DynamicConnectivity, LazyList, ConstantLengthList
# from utils.proc import load_image, get_corners_from_image, transform_keypoints_from_roi, get_padded_images, seamless_merge, crop_image, selective_color_blur, get_roi_from_corners, resize_image, seamless_merge_into_roi, seamless_gradient_merge
# from utils.utils import tqdm, no_tqdm


class TransformDetails:
    def __init__(self, M, src_pts, dst_pts):
        self.M = M
        self.src_pts = src_pts
        self.dst_pts = dst_pts

    def get(self):
        return self.M, self.src_pts, self.dst_pts

    def __str__(self):
        return f"M: {self.M}, src_pts: {self.src_pts}, dst_pts: {self.dst_pts}"


class Stitcher:
    def __init__(self,
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
                 ref_image_contrib = 0.5 # Used in alpha blending
                 ):
        
        options_dict = {
            "log_level": [logging.DEBUG, logging.INFO, logging.WARNING],
            "algorithm": ["sift", "orb", "akaze"],
            "matcher": ["bf", "flann"],
            "blender": ["alpha", "gradient"],
            "type": ["affine", "perspective"]
        }

        # Initialise logger
        if log_level != 0:
            _logger = logging.getLogger('root')
            FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
            logging.basicConfig(format=FORMAT)
            # _logger.setLevel(logging.DEBUG)
            self.logger = LogWrapper(_logger)
            self.log_level = options_dict["log_level"][log_level-1]
            self.logger.logger.setLevel(self.log_level)
        else:
            self.logger.off()

        # Initialise parameters
        self.blend_processor = blend_processor
        self.output_dir = output_dir
        self.min_match_count = min_match_count
        self.lowes_ratio_threshold = lowes_ratio_threshold
        self.ref_image_contrib = ref_image_contrib
        self.type = type
        self.resize = resize
        self.consecutive_range = consecutive_range
        self.matcher = matcher
        self.logger.info(f"Initialising {options_dict['matcher'][self.matcher].upper()} matcher")
        if self.matcher == 0:
            # BF
            self.matcher_obj = cv2.BFMatcher()
        elif self.matcher == 1:
            # FLANN
            self.matcher_obj = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        self.algorithm = algorithm
        self.logger.info(f"Initialising {options_dict['algorithm'][self.algorithm].upper()} algorithm")
        if self.algorithm == 0:
            # SIFT
            self.algorithm_obj = cv2.SIFT_create()
        elif self.algorithm == 1:
            # ORB
            self.algorithm_obj = cv2.ORB_create()
        elif self.algorithm == 2:
            # AKAZE
            self.algorithm_obj = cv2.AKAZE_create()
        self.logger.info(f"Using {options_dict['type'][self.type].upper()} type stitching")
        self.blender = blender
        self.logger.info(f"Using {options_dict['blender'][self.blender].upper()} blender")


    def get_matches(self, descriptors_1, descriptors_2, k = 2):
        if descriptors_1 is None or descriptors_2 is None:
            return list()
        if self.matcher == 1: # FLANN
            if descriptors_1.dtype != np.float32:
                descriptors_1 = descriptors_1.astype(np.float32)
            if descriptors_2.dtype != np.float32:
                descriptors_2 = descriptors_2.astype(np.float32)
        matches = self.matcher_obj.knnMatch(descriptors_1,descriptors_2,k=k)
        return matches
    
    def filter_matches(self, matches):
        good = list()
        for match in matches:
            if len(match) > 1:
                if match[0].distance < self.lowes_ratio_threshold*match[1].distance:
                    good.append([match[0]])
            else:
                good.append([match[0]])
        return good
    
    def get_pts(self, src_kp, dst_kp, good_matches):
        src_pts = np.float32([src_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([dst_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        return src_pts, dst_pts

    def get_transform(self, src_pts, dst_pts):
        if self.type == 0: # Affine 
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        elif self.type == 1: # Perspective
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        return M

    def warp_and_merge(self, img, ref, M):
        if self.type == 0: # Affine 
            warped_image = cv2.warpAffine(img, M, (ref.shape[1], ref.shape[0]))
        elif self.type == 1: # Perspective
            warped_image = cv2.warpPerspective(img, M, (ref.shape[1], ref.shape[0]))
        
        start = time.time()
        if self.blender == 0:
            warped_image = seamless_merge(warped_image, ref, self.ref_image_contrib)
        elif self.blender == 1:
            warped_image = seamless_gradient_merge(warped_image, ref)
        end = time.time()
        self.logger.debug(end-start)

        return warped_image

    def detectAndComputeFromCorners(self, image, corners):
        roi_img = get_roi_from_corners(image, corners[0], corners[1])
        keypoints_roi, thisdes = self.algorithm_obj.detectAndCompute(roi_img, None)
        thiskp = transform_keypoints_from_roi(keypoints_roi, corners[0])
        return thiskp, thisdes
    
    def save_last_stitch(self, stitch, filename):
        processed_stitch = self.blend_processor(stitch)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, processed_stitch)
        self.logger.info(f"Saved last stitch to {filepath}")

    def stitch(self, arg_ref, arg_img, tf = None, corners = None):
        current_image, reference_image = get_padded_images(arg_img, arg_ref)
        current_corners = get_corners_from_image(current_image)
        if current_corners is False:          
            self.logger.warning("Could not get image corners. The input image has no non-black regions.")
            corners = None
        if tf is None:
            # Obtain tf
            if corners != None and self.consecutive_range != None:
                try:
                    thiskp, thisdes = self.detectAndComputeFromCorners(current_image, current_corners)
                    refkp, refdes = self.detectAndComputeFromCorners(reference_image, corners)
                except Exception as e:
                    self.logger.warning("Error in detecting keypoints from corners. Computing on whole image. Exception: ",str(e))
                    corners = None
            if corners == None or self.consecutive_range == None or thisdes is None:
                thiskp, thisdes = self.algorithm_obj.detectAndCompute(current_image, None)
                refkp, refdes = self.algorithm_obj.detectAndCompute(reference_image, None)

            if refdes is None:
                self.logger.warning(f"Key points not found in ref. Moving to next blend.")
                return False # Not enough keypoints in ref
            
            matches = self.get_matches(thisdes, refdes, 2)

            good_matches = self.filter_matches(matches)

            if len(good_matches)>=self.min_match_count:
                # matched_once = True
                self.logger.debug( "Enough matches - {}/{}".format(len(good_matches), self.min_match_count))
                src_pts, dst_pts = self.get_pts(thiskp, refkp, good_matches)
                M = self.get_transform(src_pts, dst_pts)
                tf = TransformDetails(M, src_pts, dst_pts)
            else:
                self.logger.debug( "Cannot blend. Not enough matches found - {}/{}".format(len(good_matches), self.min_match_count) )
                return None # Not enough matches
        else:
            M, src_pts, dst_pts = tf.get()

        reference_image = self.warp_and_merge(current_image, reference_image, M)

        if self.consecutive_range != None:
            corners = get_corners_from_image(reference_image)                            
            if corners is False:          
                self.logger.warning("Could not get image corners. The merged image has no non-black regions.")
                corners = None

        reference_image = crop_image(reference_image) # Removing padding
        if reference_image is False:
            return None

        return reference_image, corners, tf


class ArbitraryStitcher(Stitcher):
    def __init__(self, images = None, filepaths = None, with_tqdm = True, **kwargs):
        super().__init__(**kwargs)

        # Initialise images
        if images != None:
            self.images = images
        elif filepaths != None:
            self.filepaths = filepaths
            self.logger.debug("Input filepaths:", filepaths)
            self.filepaths = [file for file in self.filepaths if file[-4:]==".png" or file[-4:]==".jpg"]

            self.filepaths = [p[1] for p in enumerate(self.filepaths) if p[0]%1==0]
            
            self.images = LazyList(lambda index: load_image(self.filepaths[index], to_resize=self.resize), len(self.filepaths))
        else:
            raise ValueError("No images specified")
        
        if with_tqdm == False:
            self.tqdm = no_tqdm
        elif with_tqdm == True:
            self.tqdm = tqdm
        else:
            self.tqdm = with_tqdm

    def update_all_keypoints_and_descriptors(self):
        kp = list()
        des = list()
        emptyindexes = list()
        for i in range(0, len(self.images)):
            thiskp, thisdes = self.algorithm_obj.detectAndCompute(self.images[i], None)
            if thisdes is None:
                emptyindexes.append(i)
            kp.append(thiskp)
            des.append(thisdes)

        # Dropping images with no keypoints
        for index in sorted(emptyindexes, reverse=True):
            # del images[index]
            self.images.length -= 1
            del kp[index]
            del des[index]
            del self.filepaths[index]

        self.kp = kp
        self.des = des

    def get_collections(self):
        self.logger.info("Calculating collections")
        connections = DynamicConnectivity(len(self.images))
        self.update_all_keypoints_and_descriptors()
        for i in self.tqdm(iterable=range(0, len(self.images))):
            for j in range(0, len(self.images)):
                if j==i:
                    continue
                if self.consecutive_range != None:
                    if abs(i-j) > self.consecutive_range:
                        continue
                matches = self.get_matches(self.des[i], self.des[j], 2)                        
                good_matches = self.filter_matches(matches)
                if len(good_matches) >= self.min_match_count:
                    self.logger.debug( "{}->{}, Matches found - {}/{}".format(i,j,len(good_matches), self.min_match_count) )
                    connections.union(i, j)
                else:
                    self.logger.debug( "{}->{}, Not enough matches found - {}/{}".format(i,j,len(good_matches), self.min_match_count) )
        collections = connections.get_connected_components()
        collections = sorted(collections, key=len, reverse=True)
        self.logger.debug(collections)
        self.logger.info(len(collections), "possible blends found")
        self.connections = connections
        return collections

    def stitch_collections(self, collections):
        self.logger.info("Stitching images")

        if self.tqdm != False:
            progress_bar = tqdm(total=len(self.images))

        unblended_collections = copy.deepcopy(collections)
        blended_collections = list()

        i = 0
        while True:
            if i >= len(unblended_collections):
                break
            unblended_image_group = unblended_collections[i]
            unblended_image_indexes = copy.deepcopy(unblended_image_group)
            if len(unblended_image_indexes) < 1:
                self.logger.warning("No images found in unblended group. Moving to next blend.","\n", unblended_image_group,"\n", unblended_collections), 
                continue
            ref = unblended_image_indexes[0]
            reference_image = self.images[ref]
            unblended_image_indexes.remove(ref)
            
            key_points_broken = False
            
            while len(unblended_image_indexes)!=0:
                self.logger.debug(unblended_image_indexes)
                matched_once = False
                remove_indexes = list()
                corners = None
                for k in unblended_image_indexes:
                    result = self.stitch(reference_image, self.images[k], corners = corners)
                    if result is False:
                        unblended_collections.append(unblended_image_indexes[k:])
                        key_points_broken = True
                        corners = None
                        break
                    elif result is None:
                        corners = None
                    else:
                        reference_image, corners, tf = result
                        matched_once = True
                        remove_indexes.append(k)
                        progress_bar.update(1)

                for r in remove_indexes:
                    unblended_image_indexes.remove(r)            
                    
                if matched_once == False:
                    # reference_image = crop_image(reference_image)
                    break

                if key_points_broken == True:
                    break

            # reference_image = cv2.medianBlur(reference_image, 3)     # Removing pepper noise developed during bitwise OR
            # reference_image = selective_color_blur(reference_image, 50, 21)
            
            # blended_collections.append(reference_image)
            self.save_last_stitch(reference_image, f"blend_{i}.png")

            if self.tqdm != False:
                progress_bar.update(1)
            i+=1

        if self.tqdm != False:
            progress_bar.close()
    
        return blended_collections  


class ConsecutiveStitcher(Stitcher):
    def __init__(self, 
                 ref_image_contrib = 0.2,
                consecutive_range = 1,
                backup_interval = False,
                consecutive_volatility_threshold = 4,
                **kwargs):
        super().__init__(**kwargs)
        self.consecutive_volatility_threshold = consecutive_volatility_threshold
        self.ref_image_contrib = ref_image_contrib
        self.stitch_count = 0
        self.consecutive_range = consecutive_range
        self.refs = ConstantLengthList(self.consecutive_range)
        self.refs.append(None)
        self.corners = None
        self.image_count = 0
        self.current_image_count = 0
        self.backup_interval = backup_interval

    def save_last_stitch(self):
        return super().save_last_stitch(self.refs[-1], f"blend_{self.stitch_count}.png")
    
    def save_and_reset(self):
        self.save_last_stitch()
        self.refs.append(None)
        self.stitch_count += 1
        self.corners = None
        self.current_image_count = 0

    def finalise_current_image(self, result):
        self.refs[-1], self.corners, tf = result
        self.image_count+=1
        self.current_image_count+=1
        if self.backup_interval != False:
            if self.image_count%self.backup_interval == 0:
                self.save_last_stitch()
        return tf

    def stitch_consecutive(self, input_image, tf = None):
        self.logger.info(f"Stitch {self.stitch_count + 1}: Stitching image {self.image_count}")
        image = resize_image(input_image, self.resize)
        # print(self.refs)
        if self.refs[-1] is None:
            self.refs[-1] = image
        else:
            result = self.stitch(self.refs[-1], image, corners = self.corners, tf=tf)
            if result in [False, None]:
                if self.current_image_count < self.consecutive_volatility_threshold:
                    for exref in reversed(self.refs[:-1]):
                        result = self.stitch(exref, image, corners = self.corners, tf=tf)   
                        if result not in [False, None]:
                            tf = self.finalise_current_image(result)
                            return tf
                self.save_and_reset() 
            else:
                tf = self.finalise_current_image(result)
                return tf
                

        return tf

