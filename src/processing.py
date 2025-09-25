import numpy as np
import cv2
from skimage.exposure import adjust_gamma
import tensorflow as tf

def preprocess(image):
    """
    Preprocesses an input image by adjusting its gamma based on luminance and applying color thresholding in the HLS color space.

    Args:
        image: Input image in RGB format.

    Returns:
        Processed image with gamma correction and color masking applied.

    The function performs the following steps:
        1. Converts the input image to HLS color space and calculates the mean luminance.
        2. Sets gamma correction and HLS threshold values based on luminance.
        3. Applies gamma correction to the image.
        4. Converts the gamma-corrected image to HLS color space.
        5. Creates a mask using the specified HLS thresholds.
        6. Applies the mask to the gamma-corrected image to extract relevant regions.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    mean_luminance = np.mean(hls[:,:,2])
        
    if mean_luminance < 50:
        gamma = 0.7
        lower = np.array([40, 0, 50])
        upper = np.array([180, 170, 255])
            
    elif mean_luminance > 85:
        gamma = 1.2
        lower = np.array([40, 0, 90])
        upper = np.array([180, 170, 255])
            
    else:
        gamma = 0.8
        lower = np.array([40, 0, 70])
        upper = np.array([180, 170, 255])
            
    image_gamma = adjust_gamma(image, gamma)
    image_hls = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(image_hls, lower, upper)
    processed_image = cv2.bitwise_and(image_gamma, image_gamma, mask=mask)

    return processed_image

def NMS(bboxes, scores, iou_threshold):
    """
    Applies Non-Maximum Suppression (NMS) to a set of bounding boxes and their associated scores.

    Args:
        bboxes: Array of bounding boxes with shape (num_boxes, 4), 
            where each box is represented as [ymin, xmin, ymax, xmax].
        scores: Array of confidence scores for each bounding box.

    Returns:
        np.ndarray: Array of selected bounding boxes after NMS suppression.
    """
    converted_bboxes = [(y-h/2, x-w/2, y+h/2, x+w/2) for x,y,w,h in bboxes]

    idxs_of_selected_bboxes = tf.image.non_max_suppression(boxes=converted_bboxes,
                                                           scores=scores,
                                                           max_output_size=len(bboxes),
                                                           iou_threshold=iou_threshold)
    
    selected_bboxes = [bboxes[idx] for idx in idxs_of_selected_bboxes]

    return np.array(selected_bboxes)

    
    