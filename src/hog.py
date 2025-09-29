import pandas as pd
import numpy as np
import math
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, 
    recall_score
)
from config import IoU
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time as t

class HOG_Detector:
    def __init__(self, 
                 base_window_size=(64,64), 
                 window_sizes=((32,32), (64,64), (96,96), (128,128)), 
                 orientations=9, 
                 pixels_per_cell=(8, 8), 
                 cells_per_block=(2, 2), 
                 name='hog'):
        
        self.base_window_size = base_window_size
        self.window_sizes = window_sizes
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.name = name

        self.block_size = (cells_per_block[0] * pixels_per_cell[0],
                           cells_per_block[1] * pixels_per_cell[1])
        
        self.blocks_per_window = (math.floor((self.base_window_size[0] - self.block_size[0])/self.pixels_per_cell[0]) + 1,
                                  math.floor((self.base_window_size[1] - self.block_size[1])/self.pixels_per_cell[1]) + 1)

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
                width = self.base_window_size[0] 
                height = self.base_window_size[1]
                x_center = (j+1) * width/2 
                y_center = (i+1) * height/2

                bboxes.append([x_center, y_center, width, height])
                features.append(feature_vector)

        return (np.array(features), np.array(bboxes))

    def get_IoUs(self, gt_bboxes, rois):
        """
        For each ROI, find the ground truth box with the highest IoU.

        Args:
            gt_bboxes: ground truth bounding boxes
            rois: regions of interest extracted from sliding window

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
        
        zoom_factor > 1.0 -> zoom in (crop + resize)
        zoom_factor < 1.0 -> zoom out (padding + resize)
        zoom_factor = 1.0 -> no changes

        Args:
            image - image
            zoom_factor - zooming factor

        Returns zoomed image of the same size.
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
    
    def scale_bboxes(self, image_size, bboxes, scale_factor):
        """
        Scale bounding boxes according to the zoom factor applied to the image.

        Args:
            image_size - tuple of image size
            bboxes - bounding boxes (gt_bboxes / rois) to be scaled
            scale_factor - zooming factor used in zoom_image function

        Returns scaled bounding boxes clipped to image boundaries.
        """
        scaled_bboxes = bboxes.copy()

        H_orig, W_orig = image_size[:2]
        H_crop, W_crop = image_size[:2]

        # scaling bboxes to zoomed image
        scaled_bboxes = np.multiply(bboxes, scale_factor)

        # offset crop
        x_offset = (W_orig * scale_factor - W_crop) / 2
        y_offset = (H_orig * scale_factor - H_crop) / 2
        scaled_bboxes[:, 0] -= x_offset
        scaled_bboxes[:, 1] -= y_offset

        # clip bboxes to image boundaries
        scaled_bboxes[:, 0] = np.clip(scaled_bboxes[:, 0], 0, W_crop)
        scaled_bboxes[:, 1] = np.clip(scaled_bboxes[:, 1], 0, H_crop)

        return scaled_bboxes
    
    def rescale_bboxes(self, image_size, bboxes, scale_factor):
        """
        Rescale bounding boxes back to the original image size.

        Args:
            image_size - tuple of image size
            bboxes - bounding boxes (gt_bboxes / rois) to be scaled
            scale_factor - zooming factor used in zoom_image function

        Returns rescaled bounding boxes.
        """
        rescaled_bboxes = bboxes.copy()    

        H_orig, W_orig = image_size[:2]
        H_crop, W_crop = image_size[:2]

        # rescaling bbox to original image size
        x_offset = (W_orig * scale_factor - W_crop) / 2
        y_offset = (H_orig * scale_factor - H_crop) / 2

        rescaled_bboxes[:, 0] = (rescaled_bboxes[:, 0] + x_offset) / scale_factor
        rescaled_bboxes[:, 1] = (rescaled_bboxes[:, 1] + y_offset) / scale_factor
        rescaled_bboxes[:, 2] = rescaled_bboxes[:, 2] / scale_factor
        rescaled_bboxes[:, 3] = rescaled_bboxes[:, 3] / scale_factor

        # clip bboxes to image boundaries
        rescaled_bboxes[:, 0] = np.clip(rescaled_bboxes[:, 0], 0, W_crop)
        rescaled_bboxes[:, 1] = np.clip(rescaled_bboxes[:, 1], 0, H_crop)

        return rescaled_bboxes

    def extract_data(self, images, annotations, max_neg_samples):
        """
        Extract HOG features and corresponding labels from images and annotations.

        Args:
            images - images used to extract feature vectors
            annotations - annotations of ground truth bounding boxes
            max_neg_samples - parameter used to control number of negative samples 

        Returns feature vectors and labels.
        """
        X_img = []
        Y_img = []
        X_bboxes = []
        Y_bboxes = []
        neg_samples_cnt = 0

        for img, gt_bboxes in tqdm(zip(images, annotations), desc="Extracting data...", total=len(images)):
            
            #selected_rois = []
            # Avoid issues with empty annotations and ensure correct shape
            gt_bboxes = np.array(gt_bboxes, dtype=float).reshape(-1, 4)

            # Multi-scale approach
            for window_size in self.window_sizes:
                zoom_factor = self.base_window_size[0] / window_size[0]

                zoomed_img = self.zoom_image(img, zoom_factor)
                scaled_bboxes = self.scale_bboxes(image_size=img.shape,
                                                  bboxes=gt_bboxes,
                                                  scale_factor=zoom_factor)

                # Extract HOG features
                hog_features = self.extract_hog(zoomed_img)
                feature_vectors, rois = self.sliding_window(hog_features)

                ious, matched_bboxes = self.get_IoUs(gt_bboxes=scaled_bboxes, 
                                                     rois=rois)

                for idx, iou in enumerate(ious):
                    if iou >= 0.4:
                        Y_img.append(1)
                        X_img.append(feature_vectors[idx])
                        # Rescale bbox coordinates back to original image size
                        bbox = rois[idx].reshape(-1, 4)
                        rescaled_bbox = self.rescale_bboxes(image_size=img.shape,
                                                            bboxes=bbox,
                                                            scale_factor=zoom_factor)
                        X_bboxes.append(rescaled_bbox.ravel())
                        Y_bboxes.append(matched_bboxes[idx])
                        #selected_rois.append(rescaled_bbox.ravel())

                    elif iou <= 0.2 and neg_samples_cnt < max_neg_samples:
                        Y_img.append(0)
                        X_img.append(feature_vectors[idx])
                        bbox = rois[idx].reshape(-1, 4)
                        rescaled_bbox = self.rescale_bboxes(image_size=img.shape,
                                                            bboxes=bbox,
                                                            scale_factor=zoom_factor)
                        X_bboxes.append(rescaled_bbox.ravel())
                        Y_bboxes.append([0,0,0,0])
                        neg_samples_cnt += 1

        return np.array(X_img), np.array(Y_img), np.array(X_bboxes), np.array(Y_bboxes)

    def train(self, images, annotations, max_neg_samples):
        """
        Train a classifier using HOG features extracted from images and bounding boxes.

        Args:
            images - images used to train model
            annotations - ground truth bounding boxes

        Saving model as hog_svm_model.pkl in models directory.
        """
        X_img, Y_img, X_bboxes, Y_bboxes = self.extract_data(images, annotations, max_neg_samples)

        # Standardize features
        scaler = StandardScaler()
        X_scaled_img = scaler.fit_transform(X_img)
        with open(f'models/{self.name}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Positive samples: {np.sum(Y_img==1)}, Negative samples: {np.sum(Y_img==0)} \n")

        # Train a linear SVM classifier
        print("Training model...")
        model = SVC(kernel='rbf', C=10, probability=True, cache_size=2000, random_state=42)
        model.fit(X_scaled_img, Y_img)
        
        # Evaluate model on training data
        Y_pred = model.predict(X_scaled_img)
        train_precision = precision_score(y_true=Y_img, y_pred=Y_pred) * 100
        train_recall = recall_score(y_true=Y_img, y_pred=Y_pred) * 100

        ious = []
        for x_bbox, y_bbox in zip(X_bboxes, Y_bboxes):
            if np.all(y_bbox != 0):
                ious.append(IoU(box1=x_bbox, box2=y_bbox))

        avg_iou = np.average(ious)

        # Save the trained model
        with open(f'models/{self.name}_svm_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        print(f"Training completed. Evaluation on training data: (Precision: {train_precision:.2f}%), (Recall: {train_recall:.2f}%), {avg_iou:.2f} avg. IoU")
        print(f"Model saved as 'models/{self.name}_svm_model.pkl'")

    def evaluate(self, images, annotations, pred_threshold):
        """
        Evaluate the trained model on a validation set.

        Args:
            images - images used to create validation set
            annotations - ground truth bounding boxes
            pred_threshold - threshold used to control accuracy % of prediction

        Printing evaluation score.
        """

        print('Evaluating model...')

        X_img = []
        Y_img = []
        X_bboxes = []
        Y_bboxes = []
        IOUs = []

        for img, gt_bboxes in tqdm(zip(images, annotations), desc="Extracting data...", total=len(images)):
            
            # Avoid issues with empty annotations and ensure correct shape
            gt_bboxes = np.array(gt_bboxes, dtype=float).reshape(-1, 4)

            # Multi-scale approach
            for window_size in self.window_sizes:
                zoom_factor = self.base_window_size[0] / window_size[0]

                zoomed_img = self.zoom_image(img, zoom_factor)
                scaled_bboxes = self.scale_bboxes(image_size=img.shape,
                                                  bboxes=gt_bboxes,
                                                  scale_factor=zoom_factor)

                # Extract HOG features
                hog_features = self.extract_hog(zoomed_img)
                feature_vectors, rois = self.sliding_window(hog_features)

                img_ious, matched_bboxes = self.get_IoUs(gt_bboxes=scaled_bboxes, 
                                                     rois=rois)

                for idx, iou in enumerate(img_ious):
                    if iou >= 0.4:
                        Y_img.append(1)
                        Y_bboxes.append(matched_bboxes[idx])

                    else:
                        Y_img.append(0)
                        Y_bboxes.append([0,0,0,0])
                    
                    X_img.append(feature_vectors[idx])
                    bbox = rois[idx].reshape(-1, 4)
                    # Rescale bbox coordinates back to original image size
                    rescaled_bbox = self.rescale_bboxes(image_size=img.shape,
                                                        bboxes=bbox,
                                                        scale_factor=zoom_factor)
                    X_bboxes.append(rescaled_bbox)
                    IOUs.append(iou)


        model = pickle.load(open(f'models/{self.name}_svm_model.pkl', 'rb'))

        # Standardize features
        scaler = pickle.load(open(f'models/{self.name}_scaler.pkl', 'rb'))
        X_scaled_img = scaler.transform(X_img)
        
        # Evaluate 
        Y_pred = model.predict(X_scaled_img)
        idxs = np.where(Y_pred >= pred_threshold)[0]
        avg_iou = np.average(np.array(IOUs)[idxs])

        val_precision = precision_score(y_true=Y_img, y_pred=Y_pred) * 100
        val_recall = recall_score(y_true=Y_img, y_pred=Y_pred) * 100
        print(f"Evaluated metrics: (Precision: {val_precision:.2f}%), (Recall: {val_recall:.2f}%), {avg_iou:.2f} avg. IoU")

    def predict(self, images, threshold):
        """
        Predict bounding boxes in the given images.

        Args:
            images - images
            threshold - threshold used to control accuracy % of prediction

        Returns bounding boxes that conatins object.
        """

        scaler = pickle.load(open(f'models/{self.name}_scaler.pkl', 'rb'))
        model = pickle.load(open(f'models/{self.name}_svm_model.pkl', 'rb'))

        final_bboxes = []
        final_scores = []
        times = []

        for img in tqdm(images, desc="Predicting on test images...", total=len(images)):
            
            start_time = t.time()

            for window_size in self.window_sizes:
                zoom_factor = self.base_window_size[0] / window_size[0]
                zoomed_img = self.zoom_image(img, zoom_factor)
                
                # Extract HOG features
                hog_features = self.extract_hog(zoomed_img)
                feature_vectors, bboxes = self.sliding_window(hog_features)

                transformed_bboxes = self.rescale_bboxes(image_size=img.shape,
                                                         bboxes=bboxes,
                                                         scale_factor=zoom_factor)


            scaled_feature_vectors = scaler.transform(feature_vectors)
            predictions = model.predict_proba(scaled_feature_vectors)[:, 1]
            idxs = np.where(predictions >= threshold)[0]

            selected_bboxes = transformed_bboxes[idxs]
            selected_scores = predictions[idxs]

            end_time = t.time()

            final_bboxes.append(selected_bboxes)
            final_scores.append(selected_scores)
            times.append(round(end_time - start_time, 2))

        return np.array(final_bboxes), np.array(final_scores), np.average(times)