import os
import pandas as pd
import numpy as np
import cv2

from config import (
    SKIMMIA_IMAGES_TRAIN, SKIMMIA_ANNOTATIONS_TRAIN,
    VISEM_IMAGES_TRAIN, VISEM_ANNOTATIONS_TRAIN,
)

def load_data(images_dir, annotations_file):
    """ 
    Load images and annotations from the specified directories and files.
    """
    annotations = pd.read_csv(annotations_file)
    image_files = set(os.listdir(images_dir))

    images = []

    for image in image_files:
        image_path = os.path.join(images_dir, image)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file {image} not found in {images_dir}")
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    return images, annotations

def iou(box1, box2):
    """
    Compute Intersection Over Union (IoU) between two bounding boxes
    in (x_center, y_center, w, h) format.
    """
    # Convert (x_center, y_center, w, h) to (x1, y1, x2, y2)
    def to_corners(box):
        x_center, y_center, w, h = box
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = to_corners(box1)
    x1_2, y1_2, x2_2, y2_2 = to_corners(box2)

    # Intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Areas of each box
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Union area
    union_area = box1_area + box2_area - inter_area

    # IoU
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def main():
    skimmia_images, skimmia_annotations = load_data(SKIMMIA_IMAGES_TRAIN, SKIMMIA_ANNOTATIONS_TRAIN)
    visem_images, visem_annotations = load_data(VISEM_IMAGES_TRAIN, VISEM_ANNOTATIONS_TRAIN)

    print(f"Loaded {len(skimmia_images)} training images and {skimmia_annotations.shape[0]} annotations from Skimmia dataset.") 
    print(f"Loaded {len(visem_images)} training images and {visem_annotations.shape[0]} annotations from Visem dataset.")

if __name__ == '__main__':
    main()