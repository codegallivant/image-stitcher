import numpy as np
import cv2
import os
import copy
import glob
import time
import gc
import logging
from utils.logger import LogWrapper
from utils.structs import DynamicConnectivity, LazyList
from utils.proc import load_image, get_corners_from_image, transform_keypoints_from_roi, getpaddedimg, seamless_merge, crop_image, selective_color_blur, get_roi_from_corners, resize_image
from utils.utils import tqdm, no_tqdm



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
                 ref_image_contrib = 0.5
                 ):
        
        options_dict = {
            "log_level": [logging.DEBUG, logging.INFO, logging.WARNING],
            "algorithm": ["sift", "orb", "akaze"],
            "matcher": ["bf", "flann"],
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
        self.output_dir = output_dir
        self.min_match_count = min_match_count
        self.lowes_ratio_threshold = lowes_ratio_threshold
        self.ref_image_contrib = ref_image_contrib
        self.type = type
        self.resize = resize
        self.matcher = matcher
        self.logger.info(f"Initialising {options_dict["matcher"][self.matcher].upper()} matcher")
        if self.matcher == 0:
            # BF
            self.matcher_obj = cv2.BFMatcher()
        elif self.matcher == 1:
            # FLANN
            self.matcher_obj = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        self.algorithm = algorithm
        self.logger.info(f"Initialising {options_dict["algorithm"][self.algorithm].upper()} algorithm")
        if self.algorithm == 0:
            # SIFT
            self.algorithm_obj = cv2.SIFT_create()
        elif self.algorithm == 1:
            # ORB
            self.algorithm_obj = cv2.ORB_create()
        elif self.algorithm == 2:
            # AKAZE
            self.algorithm_obj = cv2.AKAZE_create()

        self.logger.info(f"Using {options_dict["type"][self.type].upper()} type stitching")


    def get_matches(self, descriptors_1, descriptors_2, k):
        if self.matcher == 1: # FLANN
            if descriptors_1.dtype != np.float32:
                descriptors_1 = descriptors_1.astype(np.float32)
            if descriptors_2.dtype != np.float32:
                descriptors_2 = descriptors_2.astype(np.float32)
        matches = self.matcher_obj.knnMatch(descriptors_1,descriptors_2,k=2)
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
    
    def warp(self, current_image, reference_image, src_pts, dst_pts):
        if self.type == 0: # Affine 
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            warped_image = cv2.warpAffine(current_image, M, (reference_image.shape[1], reference_image.shape[0]))
        elif self.type == 1: # Perspective
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            warped_image = cv2.warpPerspective(current_image, M, (reference_image.shape[1], reference_image.shape[0]))
        return warped_image, M

    def detectAndComputeFromCorners(self, image, corners):
        roi_img = get_roi_from_corners(image, corners[0], corners[1])
        keypoints_roi, thisdes = self.algorithm_obj.detectAndCompute(roi_img, None)
        thiskp = transform_keypoints_from_roi(keypoints_roi, corners[0])
        return thiskp, thisdes


class ArbitraryStitcher(Stitcher):
    def __init__(self, images = None, filepaths = None, consecutive_range = None, with_tqdm = True, **kwargs):
        super().__init__(**kwargs)
        self.consecutive_range = consecutive_range

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

        image_shapes = [image.shape for image in self.images]

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
                    new_image_width = reference_image.shape[1] + (2*image_shapes[k][1])
                    new_image_height = reference_image.shape[0] + (2*image_shapes[k][0])
                    reference_image = getpaddedimg(new_image_height, new_image_width, reference_image.shape[1], reference_image.shape[0], reference_image)
                    current_image = getpaddedimg(new_image_height, new_image_width, image_shapes[k][1], image_shapes[k][0], self.images[k])
                    current_corners = get_corners_from_image(current_image)

                    start = time.time()
                    if corners != None and self.consecutive_range != None:
                        try:
                            thiskp, thisdes = self.detectAndComputeFromCorners(current_image, current_corners)
                            refkp, refdes = self.detectAndComputeFromCorners(reference_image, corners)
                        except Exception as e:
                            self.logger.warning("Error in detecting keypoints from corners. Computing on whole image. Exception: ",str(e))
                            corners = None
                    if corners == None or self.consecutive_range == None:
                        thiskp, thisdes = self.algorithm_obj.detectAndCompute(current_image, None)
                        refkp, refdes = self.algorithm_obj.detectAndCompute(reference_image, None)
        
                    end = time.time()
                    self.logger.debug("0", end-start)

                    if refdes is None:
                        self.logger.warning(f"{i} {k} Key points not found in ref. Moving to next blend.")
                        unblended_collections.append(unblended_image_indexes[k:])
                        key_points_broken = True
                        break
                        
                    # print(thisdes.dtype, refdes.dtype)
                    # matches = bf.knnMatch(thisdes,refdes,k=2)
                    start = time.time()
                    matches = self.get_matches(thisdes, refdes, 2)
                    end = time.time()
                    self.logger.debug("1", end-start)

                    start = time.time()
                    good_matches = self.filter_matches(matches)
                    end = time.time()
                    self.logger.debug("2", end-start)

                    if len(good_matches)>=self.min_match_count:
                        matched_once = True
                        start = time.time()
                        src_pts = np.float32([thiskp[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                        dst_pts = np.float32([refkp[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                        end = time.time()
                        self.logger.debug("3", end-start)
                        start = time.time()
                        warped_image, M = self.warp(current_image, reference_image, src_pts, dst_pts)
                        end = time.time()
                        self.logger.debug("4", end-start)

                        if self.consecutive_range != None:
                            try:
                                corners = get_corners_from_image(warped_image)                
                            except Exception as e:
                                self.logger.warning("Error in getting corners. Exception:",str(e))
                                corners = None

                        start = time.time()
                        reference_image = seamless_merge(warped_image, reference_image, self.ref_image_contrib)
                        end = time.time()
                        self.logger.debug("6", end-start)
                                        
                        remove_indexes.append(k)
                        self.logger.debug( "Blend {}/{} ({}/{} blended): {}->ref blended, enough matches - {}/{}".format(i+1,len(unblended_collections),len(unblended_image_group)-len(unblended_image_indexes)+len(remove_indexes),len(unblended_image_group),k,len(good_matches), self.min_match_count) )
                        progress_bar.update(1)
                    else:
                        self.logger.debug( "Blend {}/{} ({}/{} blended): {}->ref, Not enough matches found - {}/{}".format(i+1,len(unblended_collections),len(unblended_image_group)-len(unblended_image_indexes)+len(remove_indexes),len(unblended_image_group),k,len(good_matches), self.min_match_count) )
                        corners = None

                    reference_image = crop_image(reference_image) # Removing padding

                for r in remove_indexes:
                    unblended_image_indexes.remove(r)            
                    
                if matched_once == False:
                    # reference_image = crop_image(reference_image)
                    break

                if key_points_broken == True:
                    break

            # reference_image = cv2.medianBlur(reference_image, 3)     # Removing pepper noise developed during bitwise OR
            reference_image = selective_color_blur(reference_image, 50, 21)
            blended_collections.append(reference_image)
            if self.tqdm != False:
                progress_bar.update(1)
            i+=1

        if self.tqdm != False:
            progress_bar.close()
    
        return blended_collections  


class ConsecutiveStitcher(Stitcher):
    def __init__(self, ref_image_contrib = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.ref_image_contrib = ref_image_contrib
        self.refs = [None]
        self.corners = None

    def stitch(self, input_image):
        self.logger.info("Stitching image")
        image = resize_image(input_image, self.resize)
        # print(self.refs)
        if self.refs[-1] is None:
            self.refs[-1] = image
        else:
            reference_image = self.refs[-1]
            new_image_width = reference_image.shape[1] + (2*image.shape[1])
            new_image_height = reference_image.shape[0] + (2*image.shape[0])
            reference_image = getpaddedimg(new_image_height, new_image_width, reference_image.shape[1], reference_image.shape[0], reference_image)
            current_image = getpaddedimg(new_image_height, new_image_width, image.shape[1], image.shape[0], image)
            current_corners = get_corners_from_image(current_image)

            if self.corners != None:
                try:
                    thiskp, thisdes = self.detectAndComputeFromCorners(current_image, current_corners)
                    refkp, refdes = self.detectAndComputeFromCorners(reference_image, self.corners)
                except Exception as e:
                    self.logger.warning("Error in detecting keypoints from corners. Computing on whole image. Exception: ",str(e))
                    self.corners = None
            if self.corners == None:
                thiskp, thisdes = self.algorithm_obj.detectAndCompute(current_image, None)
                refkp, refdes = self.algorithm_obj.detectAndCompute(reference_image, None)

            if refdes is None:
                self.logger.warning(f"Key points not found in ref. Moving to next blend.")
                self.refs.append(image)
                return None
                
            matches = self.get_matches(thisdes, refdes, 2)
            good_matches = self.filter_matches(matches)
            
            if len(good_matches)>=self.min_match_count:
                src_pts = np.float32([thiskp[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([refkp[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                warped_image, M = self.warp(current_image, reference_image, src_pts, dst_pts)
            
                try:
                    self.corners = get_corners_from_image(warped_image)                
                except Exception as e:
                    self.logger.warning("Error in getting corners. Exception:",str(e))
                    self.corners = None

                reference_image = seamless_merge(warped_image, reference_image, self.ref_image_contrib)
                                
                self.logger.debug( "Blended. Matches - {}/{}".format(len(good_matches), self.min_match_count) )

            else:
                self.logger.debug( "Not enough matches: : {}/{}".format(len(good_matches), self.min_match_count) )
                self.corners = None
                M = None
            
            reference_image = crop_image(reference_image) # Removing padding
            self.refs[-1] = reference_image
            if M is None:
                self.refs.append(image)

            return M


