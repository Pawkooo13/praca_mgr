import os
import pandas as pd
import numpy as np
import cv2
from hog import HOG_Detector

from paths import (
    SKIMMIA_IMAGES_TRAIN, SKIMMIA_ANNOTATIONS_TRAIN,
    SKIMMIA_IMAGES_VALID, SKIMMIA_ANNOTATIONS_VALID,
    SKIMMIA_IMAGES_TEST, SKIMMIA_ANNOTATIONS_TEST,
    VISEM_IMAGES_TRAIN, VISEM_ANNOTATIONS_TRAIN,
    VISEM_IMAGES_VALID, VISEM_ANNOTATIONS_VALID,
    VISEM_IMAGES_TEST, VISEM_ANNOTATIONS_TEST
)

def load_data(images_dir, annotations_file):
    """ 
    Load images and annotations from the specified directories and files.
    Returns arrays of images and corresponding bounding boxes.
    """
    annotations = pd.read_csv(annotations_file)
    # Remove file extension from image names for consistency (skimmia dataset)
    if 'skimmia' in annotations_file:
        annotations['image'] = annotations['image'].str[:-4]

    image_files = set(os.listdir(images_dir)[:100])

    images = []
    bboxes = []

    for image in image_files:
        image_path = os.path.join(images_dir, image)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file {image} not found in {images_dir}")
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            img_name = image[:-4] # remove .jpg extension
            img_annotations = annotations.query("`image` == @img_name")
            img_bboxes = img_annotations[['x_center', 'y_center', 'width', 'height']].values.tolist()
            bboxes.append(img_bboxes)

    return np.array(images), np.array(bboxes)

def main():

    print("Loading skimmia data... \n")

    skimmia_images_train, skimmia_annotations_train = load_data(SKIMMIA_IMAGES_TRAIN, SKIMMIA_ANNOTATIONS_TRAIN)
    skimmia_images_valid, skimmia_annotations_valid = load_data(SKIMMIA_IMAGES_VALID, SKIMMIA_ANNOTATIONS_VALID)
    skimmia_images_test, skimmia_annotations_test = load_data(SKIMMIA_IMAGES_TEST, SKIMMIA_ANNOTATIONS_TEST)

    #print("Loading visem data... \n")
    #visem_images_train, visem_annotations_train = load_data(VISEM_IMAGES_TRAIN, VISEM_ANNOTATIONS_TRAIN)
    #visem_images_valid, visem_annotations_valid = load_data(VISEM_IMAGES_VALID, VISEM_ANNOTATIONS_VALID)
    #visem_images_test, visem_annotations_test = load_data(VISEM_IMAGES_TEST, VISEM_ANNOTATIONS_TEST)

    print(f"Loaded {skimmia_images_train.shape[0]} training images and {skimmia_annotations_train.shape[0]} annotations from Skimmia dataset.") 
    #print(f"Loaded {visem_images_train.shape[0]} training images and {visem_annotations_train.shape[0]} annotations from Visem dataset.")

    hog_detector = HOG_Detector()
    hog_detector.train(images=skimmia_images_train, 
                       annotations=skimmia_annotations_train)
    
    hog_detector.evaluate(images=skimmia_images_valid, 
                          annotations=skimmia_annotations_valid)
    '''
    for img, bboxes in zip(skimmia_images, skimmia_annotations):

        hog_features = hog_detector.extract_hog(img)
        print(hog_features.shape)

        feature_vectors, bboxes = hog_detector.sliding_window(hog_features)
        print(feature_vectors.shape)
        print(bboxes.shape)
        print(bboxes)

    #plot bboxes on image
    img = skimmia_images[0].copy()
    bboxes = skimmia_annotations[0]

    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    cv2.imshow('Image with bboxes', img)
    cv2.waitKey(0)
    '''

if __name__ == '__main__':
    main()