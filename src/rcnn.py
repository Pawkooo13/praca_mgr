import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras import (
    Sequential, 
    Input
)
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomFlip,
    RandomTranslation,
    RandomZoom,
    Rescaling
)
from tensorflow.keras.layers import (
    Dense, 
    Conv2D, 
    MaxPool2D,
    Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision
from tensorflow.keras.models import load_model
from tqdm import tqdm
from config import IoU

class RCNN:
    def __init__(self, resize, cnn, name):
        self.name = name
        self.resize = resize
        self.cnn = cnn

    def get_ROIs(self, image):
        """
        Generates region proposals (ROIs) from the input image using Selective Search algorithm.

        Args:
            image: The input image for which to generate region proposals.

        Returns:
            - rois_images: A list of ROI images resized to (64, 64, 3).
            - rois_coordinates: A list of ROI coordinates, where each ROI is represented as [x, y, w, h].
        """
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSingleStrategy(k=200, sigma=0.8)
        rois_coordinates = ss.process()

        # Preallocate output arrays for efficiency
        num_rois = len(rois_coordinates)
        img_w, img_h = self.resize
        rois_images = np.empty((num_rois, img_w, img_h, 3), dtype=np.uint8)

        for i, (x, y, w, h) in enumerate(rois_coordinates):
            region = image[y:y+h, x:x+w]
            region_resized = cv2.resize(region, (img_w, img_h))
            rois_images[i] = region_resized

        rois_coords = [[x+w/2, y+h/2, w, h] for x,y,w,h in rois_coordinates]
        return rois_images, np.array(rois_coords)
    
    def get_IOUs(self, gt_bboxes, rois):
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
    
    def extract_data(self, images, annotations, max_neg_samples):
        """
        Extracts training data for an RCNN model from images and their annotations.
        This method generates positive and negative samples based on the IoU between region proposals (ROIs) 
        and ground truth bounding boxes. Positive samples are those with IoU >= 0.4, 
        and negative samples are those with IoU <= 0.2, up to a maximum specified by `max_neg_samples`.

        Args:
            images: List or array of input images.
            annotations: List of ground truth bounding boxes for each image.
            max_neg_samples: Maximum number of negative samples to extract.

        Returns:
            - Array of ROI images.
            - Array of labels (1 for positive, 0 for negative).
            - Array of ROI bounding boxes.
            - Array of target bounding boxes (ground truth for positives, zeros for negatives).
        """

        X_imgs = []
        Y_imgs = []
        X_bboxes = []
        Y_bboxes = []
        
        neg_samples_cnt = 0

        for img, gt_bboxes in tqdm(zip(images, annotations), desc="Extracting data...", total=len(images)):
            
            rois_imgs, rois_bboxes = self.get_ROIs(image=img) 

            ious, matched_bboxes = self.get_IOUs(gt_bboxes=gt_bboxes, 
                                                 rois=rois_bboxes)

            for idx, iou in enumerate(ious):
                if iou >= 0.4:
                    Y_imgs.append([1.0])
                    X_imgs.append(rois_imgs[idx])
                    X_bboxes.append(rois_bboxes[idx])
                    Y_bboxes.append(matched_bboxes[idx])
                
                elif iou <= 0.2 and neg_samples_cnt < max_neg_samples:
                    Y_imgs.append([0.0])
                    X_imgs.append(rois_imgs[idx])
                    X_bboxes.append(rois_bboxes[idx])
                    Y_bboxes.append(np.array([0, 0, 0, 0]))
                    neg_samples_cnt += 1

        return np.array(X_imgs), np.array(Y_imgs), np.array(X_bboxes), np.array(Y_bboxes)

    def train(self, images, annotations, max_neg_samples, epochs):
        """
        Trains the RCNN model using the provided images and annotations.

        Args:
            images: List of input images for training.
            annotations: Corresponding bounding boxes for the images.
            max_neg_samples: Maximum number of negative samples to extract for training.
            epochs: Number of training epochs.
            name: Name identifier for saving the model and training history.

        Returns:
            - Prints training progress and evaluation metrics.
            - Saves training history plot to 'plots/training_history_<name>.png'.
            - Saves trained model to 'models/<name>.h5'.
        """

        X_img, Y_img, X_bboxes, Y_bboxes = self.extract_data(images=images,
                                                             annotations=annotations,
                                                             max_neg_samples=max_neg_samples)
        
        print(f"Positive samples: {np.sum(Y_img==1)}, Negative samples: {np.sum(Y_img==0)} \n")

        print('Training model...')
        model = self.cnn
        training_history = model.fit(x=X_img,
                                     y=Y_img,
                                     batch_size=64,
                                     epochs=epochs,
                                     validation_split=0.3)

        # Save training history
        pd.DataFrame(training_history.history).plot()
        plot_path = os.path.join('plots', 'training_history_' + str(self.name) + '.png')
        plt.savefig(plot_path)

        # Evaluate model
        loss, precision = model.evaluate(x=X_img,
                                         y=Y_img,
                                         batch_size=64)
        ious = []
        for x_bbox, y_bbox in zip(X_bboxes, Y_bboxes):
            if np.all(y_bbox != 0):
                ious.append(IoU(box1=x_bbox, box2=y_bbox))

        avg_iou = np.average(ious)

        # Save model
        model_path = os.path.join('models', str(self.name) + '.h5')
        model.save(model_path)

        print(f"Training completed. Evaluation on training data: {precision * 100:.2f}% precision, {avg_iou:.2f} avg. IoU")
        print(f"Model saved as {model_path}")

    def evaluate(self, images, annotations, pred_threshold):
        """
         Evaluate the trained object detection model on a given dataset.

        Args:
            images: List of input images.
            annotations: List of ground truth bounding boxes.
            pred_threshold: Threshold on predicted probability to consider an ROI as positive.

        Print:
            - val_precision: Classification precision of the model.
            - avg_iou: Average IoU of predicted positive ROIs.
        """
        print('Evaluating model...')

        X_imgs = []
        Y_imgs = []
        X_bboxes = []
        Y_bboxes = []
        IOUs = []

        for img, gt_bboxes in tqdm(zip(images, annotations), desc="Extracting data...", total=len(images)):
            
            rois_imgs, rois_bboxes = self.get_ROIs(image=img) 

            rois_ious, matched_bboxes = self.get_IOUs(gt_bboxes=gt_bboxes, 
                                                      rois=rois_bboxes)

            for idx, iou in enumerate(rois_ious):
                if iou >= 0.4:
                    Y_imgs.append([1.0])
                    Y_bboxes.append(matched_bboxes[idx])
                else:
                    Y_imgs.append([0.0])
                    Y_bboxes.append(np.array([0, 0, 0, 0]))

                X_imgs.append(rois_imgs[idx])
                X_bboxes.append(rois_bboxes[idx])
                IOUs.append(iou)

        # Computing precision
        model = load_model(filepath=f'models/{self.name}.h5', 
                           compile=True)
        loss, val_precision = model.evaluate(x=np.array(X_imgs),
                                             y=np.array(Y_imgs),
                                             batch_size=64,
                                             verbose=0)

        Y_pred = model.predict(np.array(X_imgs)).flatten()
        idxs = np.where(Y_pred >= pred_threshold)[0]
        avg_iou = np.average(np.array(IOUs)[idxs])

        print(f"Evaluated precision: {val_precision*100:.2f}%, {avg_iou:.2f} avg. IoU")

    def predict(self, images, threshold):
        """
        Generates predictions for a list of images using a pre-trained model and a specified threshold.

        Args:
            images: List or array of input images to process.
            threshold: Confidence threshold for filtering predictions.

        Returns:
            - final_rois: Regions of interest (ROIs) from images that meet the threshold.
            - final_bboxes: Bounding boxes corresponding to the selected ROIs.
            - final_scores: Prediction scores for the selected ROIs.
        """

        model = load_model(filepath=f'models/{self.name}.h5', 
                           compile=True)

        final_scores = []
        final_rois = []
        final_bboxes = []

        for img in tqdm(images, desc="Predicting on test images...", total=len(images)):
            
            rois_imgs, rois_bboxes = self.get_ROIs(image=img) 

            predictions = model.predict(rois_imgs).flatten()
            idxs = np.where(predictions >= threshold)[0]

            final_scores.append(predictions[idxs])
            final_rois.append(rois_imgs[idxs])
            final_bboxes.append(rois_bboxes[idxs])

        return np.array(final_rois), np.array(final_bboxes), np.array(final_scores)