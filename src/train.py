import os
import pandas as pd
import numpy as np
import cv2

from paths import (
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

def main():
    skimmia_images, skimmia_annotations = load_data(SKIMMIA_IMAGES_TRAIN, SKIMMIA_ANNOTATIONS_TRAIN)
    visem_images, visem_annotations = load_data(VISEM_IMAGES_TRAIN, VISEM_ANNOTATIONS_TRAIN)

    print(f"Loaded {len(skimmia_images)} training images and {skimmia_annotations.shape[0]} annotations from Skimmia dataset.") 
    print(f"Loaded {len(visem_images)} training images and {visem_annotations.shape[0]} annotations from Visem dataset.")

if __name__ == '__main__':
    main()