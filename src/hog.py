import pandas as pd
import numpy as np
import cv2
from skimage.feature import hog

class HOG_Detector:
    def __init__(self, window_size=(64,64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.window_size = window_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def extract_hog(self, image):
        """
        Compute HOG features for a given image in RGB format.
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        hog_features = hog(image,
                           orientations=self.orientations,
                           pixels_per_cell=self.pixels_per_cell,
                           cells_per_block=self.cells_per_block,
                           visualize=False,
                           feature_vector=True,
                           multichannel=False)
        
        return hog_features
    
    def sliding_window(self, image):
        pass

    def train(self, images, labels):
        pass

    def predict(self, image):
        pass