import numpy as np
import cv2
from skimage.exposure import adjust_gamma

def preprocess(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    mean_luminance = np.mean(hls[:,:,2])
        
    if mean_luminance < 50:
        gamma = 0.7
        lower = np.array([30, 0, 50])
        upper = np.array([180, 170, 255])
            
    elif mean_luminance > 85:
        gamma = 1.2
        lower = np.array([30, 0, 90])
        upper = np.array([180, 170, 255])
            
    else:
        gamma = 0.8
        lower = np.array([30, 0, 70])
        upper = np.array([180, 170, 255])
            
    image_gamma = adjust_gamma(image, gamma)
    image_hls = cv2.cvtColor(image_gamma, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(image_hls, lower, upper)
    processed_image = cv2.bitwise_and(image_gamma, image_gamma, mask=mask)

    return processed_image