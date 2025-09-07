import pandas as pd
import numpy as np
import math
import cv2
from skimage.feature import hog

class HOG_Detector:
    def __init__(self, window_size=(64,64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.window_size = window_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

        self.block_size = (cells_per_block[0] * pixels_per_cell[0],
                           cells_per_block[1] * pixels_per_cell[1])
        
        self.blocks_per_window = (math.floor((self.window_size[0] - self.block_size[0])/self.pixels_per_cell[0]) + 1,
                                  math.floor((self.window_size[1] - self.block_size[1])/self.pixels_per_cell[1]) + 1)

    def extract_hog(self, image):
        """
        Compute HOG features for a given image in RGB format.

        feature_vector=False returns a vector in format:
            (blocks_col, blocks_row, n_cells_per_block, n_cells_per_block, n_orientations)

        Retruns feature vectors for each block.
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        hog_features = hog(image,
                           orientations=self.orientations,
                           pixels_per_cell=self.pixels_per_cell,
                           cells_per_block=self.cells_per_block,
                           visualize=False,
                           feature_vector=False,
                           channel_axis=None)
        
        feature_shape = (hog_features.shape[0],
                         hog_features.shape[1],
                         hog_features.shape[2] * hog_features.shape[3] * hog_features.shape[4])

        return hog_features.reshape(feature_shape)
    
    def sliding_window(self, hog_features):
        """
        Slide a window over the HOG feature map and extract features.
        
        Args:
            hog_features: HOG feature map from extract_hog
            
        Returns:
            features: Array of feature vectors for each window
            bboxes: List of coordinates (x_center, y_center, width, height) for each window
        """
        features = []
        bboxes = []

        # Get dimensions of HOG feature map
        h, w = hog_features.shape[:2]
        
        # Step size for sliding window (50% overlap)
        stride = (math.floor(self.blocks_per_window[0]//2) + 1,
                  math.floor(self.blocks_per_window[1]//2) + 1)

        for i, y in enumerate(range(stride[0], h, stride[0])):
            for j, x in enumerate(range(stride[1], w, stride[1])):
                window = hog_features[y:y + self.blocks_per_window[0],
                                      x:x + self.blocks_per_window[1]]
                
                # Extract features
                feature_vector = window.flatten()

                # Extract coordinates
                width = self.window_size[0] 
                height = self.window_size[1]
                x_center = (j+1) * width//2 
                y_center = (i+1) * height//2

                bboxes.append([x_center, y_center, width, height])
                features.append(feature_vector)

        return (np.array(features), np.array(bboxes))

    def train(self, images, labels):
        pass

    def predict(self, image):
        pass