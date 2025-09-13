import pandas as pd
import numpy as np
import math
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from config import IoU
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        
        # Reshape to get feature vectors for each block
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
                
                # Skip incomplete windows at the edges
                if window.shape[0] != self.blocks_per_window[0] or window.shape[1] != self.blocks_per_window[1]:
                    continue 

                # Extract features
                feature_vector = window.ravel()

                # Extract coordinates
                width = self.window_size[0] 
                height = self.window_size[1]
                x_center = (j+1) * width/2 
                y_center = (i+1) * height/2

                bboxes.append([x_center, y_center, width, height])
                features.append(feature_vector)

        return (np.array(features), np.array(bboxes))

    def get_IoUs(self, gt_bboxes, rois):
        """
        For each ROI, find the ground truth box with the highest IoU.
        Returns arrays of max IoUs and corresponding ground truth bboxes.
        """
        num_rois = len(rois)
        ious = np.zeros(num_rois, dtype=float)
        bboxes = np.zeros((num_rois, 4), dtype=float)

        for i, roi in enumerate(rois):
            ious_with_gt = [IoU(gt_box, roi) for gt_box in gt_bboxes]
            if ious_with_gt:
                max_idx = np.argmax(ious_with_gt)
                ious[i] = ious_with_gt[max_idx]
                bboxes[i] = gt_bboxes[max_idx]

        return ious, bboxes

    def zoom_image(self, image, zoom_factor):
        """
        Digital zoom on the image with given zoom factor. 
        Returns zoomed image of the same size.
        
        zoom_factor > 1.0 -> zoom in (crop + resize)
        zoom_factor < 1.0 -> zoom out (padding + resize)
        zoom_factor = 1.0 -> no changes
        """
        h, w = image.shape[:2]

        if zoom_factor > 1:  # zoom in
            new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            cropped = image[y1:y1+new_h, x1:x1+new_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

        elif zoom_factor < 1:  # zoom out
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # padding
            top = (h - new_h) // 2
            bottom = h - new_h - top
            left = (w - new_w) // 2
            right = w - new_w - left
            zoomed = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=[0, 0, 0])

        else:
            zoomed = image.copy()

        return zoomed

    def extract_data(self, images, annotations, max_neg_samples=1000):
        """
        Extract HOG features and corresponding labels from images and annotations.
        """
        X_data = []
        Y_data = []
        neg_samples_cnt = 0

        for img, gt_bboxes in tqdm(zip(images, annotations), desc="Extracting data", total=len(images)):
            
            selected_rois = []
            # to avoid issues with empty annotations and ensure correct shape
            gt_bboxes = np.array(gt_bboxes, dtype=float).reshape(-1, 4)

            # Multi-scale approach
            for window_size in [(32,32), (64,64), (96,96), (128,128)]:
                zoom_factor = self.window_size[0] / window_size[0]

                zoomed_img = self.zoom_image(img, zoom_factor)

                scaled_bboxes = gt_bboxes.copy()
                H_orig, W_orig = img.shape[:2]
                H_crop, W_crop = img.shape[:2]

                # scaling bboxes to zoomed image
                scaled_bboxes = np.multiply(scaled_bboxes, zoom_factor)

                # offset crop
                x_offset = (W_orig * zoom_factor - W_crop) / 2
                y_offset = (H_orig * zoom_factor - H_crop) / 2
                scaled_bboxes[:, 0] -= x_offset
                scaled_bboxes[:, 1] -= y_offset

                # clip bboxes to image boundaries
                scaled_bboxes[:, 0] = np.clip(scaled_bboxes[:, 0], 0, W_crop)
                scaled_bboxes[:, 1] = np.clip(scaled_bboxes[:, 1], 0, H_crop)

                hog_features = self.extract_hog(zoomed_img)
                feature_vectors, rois = self.sliding_window(hog_features)

                ious, matched_bboxes = self.get_IoUs(gt_bboxes=scaled_bboxes, 
                                                     rois=rois)

                for idx, iou in enumerate(ious):
                    if iou >= 0.4:
                        Y_data.append(1)
                        X_data.append(feature_vectors[idx])

                        # Rescale bbox coordinates back to original image size
                        rescaled_bbox = rois[idx].copy()
                        rescaled_bbox[0] = (rescaled_bbox[0] + x_offset) / zoom_factor
                        rescaled_bbox[1] = (rescaled_bbox[1] + y_offset) / zoom_factor
                        rescaled_bbox[2] = rescaled_bbox[2] / zoom_factor
                        rescaled_bbox[3] = rescaled_bbox[3] / zoom_factor

                        selected_rois.append(rescaled_bbox)

                    elif iou <= 0.2 and neg_samples_cnt < max_neg_samples:
                        Y_data.append(0)
                        X_data.append(feature_vectors[idx])
                        neg_samples_cnt += 1

        return np.array(X_data), np.array(Y_data)

    def train(self, images, annotations):
        """
        Train a classifier using HOG features extracted from images and bounding boxes.
        """
        X_train, Y_train = self.extract_data(images, annotations)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        with open('models/hog_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Positive samples: {np.sum(Y_train==1)}, Negative samples: {np.sum(Y_train==0)} \n")

        # Train a linear SVM classifier
        print("Training model...")
        model = SVC(kernel='linear', probability=True, cache_size=2000, random_state=42)
        model.fit(X_train, Y_train)
        
        # Evaluate model on training data
        train_accuracy = model.score(X_train, Y_train) * 100

        # Save the trained model
        with open('models/hog_svm_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        print("Model trained and saved as 'models/hog_svm_model.pkl'")
        print(f"Training completed. Evaluation on training data: {train_accuracy:.2f}% accuracy")
        
    def evaluate(self, images, annotations):
        """
        Evaluate the trained model on a validation set.
        """
        model = pickle.load(open('models/hog_svm_model.pkl', 'rb'))

        X_val, Y_val = self.extract_data(images, annotations)

        # Standardize features
        scaler = pickle.load(open('models/hog_scaler.pkl', 'rb'))
        X_val = scaler.transform(X_val)

        val_accuracy = model.score(X_val, Y_val) * 100
        print(f"Validation accuracy: {val_accuracy:.2f}%")

    def predict(self, images):
        """
        Predict bounding boxes in the given images.
        """
        model = pickle.load(open('models/hog_svm_model.pkl', 'rb'))

        results = []

        for img in images:
            # Extract HOG features and sliding window
            hog_features = self.extract_hog(img)
            feature_vectors, bboxes = self.sliding_window(hog_features)

            # Standardize features
            scaler = pickle.load(open('models/hog_scaler.pkl', 'rb'))
            feature_vectors = scaler.transform(feature_vectors)
            
            # Predict using the trained model
            predictions = model.predict_proba(feature_vectors)
            predictions = predictions[:, 1]  # Probability of positive class
            idxs = np.where(predictions >= 0.5)

            selected_bboxes = bboxes[idxs]
            results.append(selected_bboxes)

            # NMS ...

        # Plot 5 random images with detected boxes
        fig, ax = plt.subplots(1, 5, figsize=(20, 10))
        random_indices = np.random.choice(len(images), size=5, replace=False)
        for i, idx in enumerate(random_indices):
            img = images[idx].copy()
            for box in results[idx]:
                x_center, y_center, width, height = box
                top_left = (int(x_center - width/2), int(y_center - height/2))
                bottom_right = (int(x_center + width/2), int(y_center + height/2))
                cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
            ax[i].imshow(img)
            ax[i].axis('off')
        
        plt.savefig('plots/hog_detections.png')
        plt.show()

        return results