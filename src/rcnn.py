import numpy as np
import cv2
import tensorflow as tf

class RCNN:
    def __init__(self):
        pass

    def get_ROIs(self, image):
        """
        Generates region proposals (ROIs) from the input image using Selective Search algorithm.

        Args:
            image: The input image for which to generate region proposals.

        Returns:
            rois_iamges: A list of ROI images resized to (64, 64, 3).
            rois_coordinates: A list of ROI coordinates, where each ROI is represented as [x, y, w, h].
        """
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSingleStrategy(k=200, sigma=0.8)
        rois_coordinates = ss.process()

        # Preallocate output arrays for efficiency
        num_rois = len(rois_coordinates)
        rois_images = np.empty((num_rois, 64, 64, 3), dtype=np.uint8)

        for i, (x, y, w, h) in enumerate(rois_coordinates):
            region = image[y:y+h, x:x+w]
            region_resized = cv2.resize(region, (64, 64), interpolation=cv2.INTER_LINEAR)
            rois_images[i] = region_resized

        return rois_images, np.array(rois_coordinates)