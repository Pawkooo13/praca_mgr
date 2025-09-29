import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To remove CUDA logs
import pandas as pd
import numpy as np
import cv2
from hog import HOG_Detector
from cnn_skimmia import get_skimmia_cnn
from cnn_visem import get_visem_cnn
from rcnn import RCNN
from processing import (
    preprocess, 
)
from config import (
    plot_predictions, 
    average_IoU_after_NMS,
    avg_IoU_after_NMS_with_given_tsh
)
import warnings

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
    # Remove file extension from image names for consistency (Skimmia dataset)
    if 'skimmia' in annotations_file:
        annotations['image'] = annotations['image'].str[:-4]
        qry = "`image` == @img_name"
    else:
        qry = "`image` == @img_name & `class_id` == 0"

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
            img_annotations = annotations.query(qry)
            img_bboxes = img_annotations[['x_center', 'y_center', 'width', 'height']].values.tolist()
            bboxes.append(img_bboxes)

    return np.array(images), np.array(bboxes)

def main():
    warnings.filterwarnings("ignore")

    #-------------------- LOADING SKIMMIA DATASET --------------------
    
    print('\n')
    print("Loading skimmia data... \n")

    skimmia_images_train, skimmia_bboxes_train = load_data(SKIMMIA_IMAGES_TRAIN, SKIMMIA_ANNOTATIONS_TRAIN)
    print(f"Loaded {skimmia_images_train.shape[0]} training images from Skimmia dataset.") 

    skimmia_images_valid, skimmia_bboxes_valid = load_data(SKIMMIA_IMAGES_VALID, SKIMMIA_ANNOTATIONS_VALID)
    print(f"Loaded {skimmia_images_valid.shape[0]} validating images from Skimmia dataset.") 

    skimmia_images_test, skimmia_bboxes_test = load_data(SKIMMIA_IMAGES_TEST, SKIMMIA_ANNOTATIONS_TEST)
    print(f"Loaded {skimmia_images_test.shape[0]} testing images from Skimmia dataset.") 

    #-------------------- TRAINING HOG DETECTOR (SKIMMIA) --------------------
    print('\n')
    print('-------------------- HOG --------------------') 

    hog_skimmia_detector = HOG_Detector(name='hog_skimmia')

    hog_skimmia_detector.train(images=skimmia_images_train, 
                               annotations=skimmia_bboxes_train,
                               max_neg_samples=20000)
    
    hog_skimmia_detector.evaluate(images=skimmia_images_valid, 
                                  annotations=skimmia_bboxes_valid,
                                  pred_threshold=0.6)
    
    hog_skimmia_pred_bboxes, hog_skimmia_pred_scores, hog_skimmia_avg_pred_time = hog_skimmia_detector.predict(images=skimmia_images_test,
                                                                                                               threshold=0.6)

    avg_IoU_after_NMS_with_given_tsh(gt_bboxes=skimmia_bboxes_test,
                                     pred_bboxes=hog_skimmia_pred_bboxes,
                                     pred_scores=hog_skimmia_pred_scores,
                                     thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
                                     name='hog_skimmia')
    
    print(f"Average prediction time: {hog_skimmia_avg_pred_time:.2f} s")
                                                                                    
    hog_skimmia_avg_iou, hog_skimmia_selected_bboxes = average_IoU_after_NMS(gt_bboxes=skimmia_bboxes_test,
                                                                             pred_bboxes=hog_skimmia_pred_bboxes,
                                                                             pred_scores=hog_skimmia_pred_scores,
                                                                             nms_threshold=0.1)
                                       
    plot_predictions(images=skimmia_images_test[:5],
                     gt_bboxes=skimmia_bboxes_test[:5],
                     pred_bboxes=hog_skimmia_selected_bboxes[:5],
                     name='hog_skimmia_images_predictions')

    #-------------------- TRAINING RCNN MODEL (SKIMMIA) --------------------
    print('\n')
    print('-------------------- RCNN --------------------')

    cnn_skimmia = get_skimmia_cnn()

    rcnn_skimmia = RCNN(resize=(96, 96), 
                        cnn=cnn_skimmia, 
                        name='rcnn_skimmia')

    rcnn_skimmia.train(images=skimmia_images_train,
                       annotations=skimmia_bboxes_train,
                       max_neg_samples=30000,
                       epochs=80)
    
    rcnn_skimmia.evaluate(images=skimmia_images_valid,
                          annotations=skimmia_bboxes_valid,
                          pred_threshold=0.6)

    rcnn_skimmia_pred_rois, rcnn_skimmia_pred_bboxes, rcnn_skimmia_pred_scores, rcnn_skimmia_avg_pred_time = rcnn_skimmia.predict(images=skimmia_images_test, 
                                                                                                                                  threshold=0.6)

    avg_IoU_after_NMS_with_given_tsh(gt_bboxes=skimmia_bboxes_test,
                                     pred_bboxes=rcnn_skimmia_pred_bboxes,
                                     pred_scores=rcnn_skimmia_pred_scores,
                                     thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
                                     name='rcnn_skimmmia')
    
    print(f"Average prediction time: {rcnn_skimmia_avg_pred_time:.2f} s")

    rcnn_skimmia_avg_iou, rcnn_skimmia_selected_bboxes = average_IoU_after_NMS(gt_bboxes=skimmia_bboxes_test,
                                                                               pred_bboxes=rcnn_skimmia_pred_bboxes,
                                                                               pred_scores=rcnn_skimmia_pred_scores,
                                                                               nms_threshold=0.1)

    plot_predictions(images=skimmia_images_test[:5],
                     gt_bboxes=skimmia_bboxes_test[:5],
                     pred_bboxes=rcnn_skimmia_selected_bboxes[:5],
                     name='rcnn_skimmia_images_predictions')

    #-------------------- LOADING PREPROCESSED SKIMMIA IMAGES --------------------
    print('\n')
    print('Loading preprecessed skimmia images... \n')

    preprocessed_skimmia_images_train = np.array([preprocess(image) for image in skimmia_images_train])
    print(f"Loaded {preprocessed_skimmia_images_train.shape[0]} training images from Skimmia dataset.") 
    preprocessed_skimmia_images_valid = np.array([preprocess(image) for image in skimmia_images_valid])
    print(f"Loaded {preprocessed_skimmia_images_valid.shape[0]} validating images from Skimmia dataset.") 
    preprocessed_skimmia_images_test = np.array([preprocess(image) for image in skimmia_images_test])
    print(f"Loaded {preprocessed_skimmia_images_test.shape[0]} testing images from Skimmia dataset.") 
    
    #-------------------- TRAINING HOG DETECTOR (SKIMMIA PREPROCESSED) --------------------
    print('\n')
    print('-------------------- HOG --------------------') 

    hog_pre_skimmia_detector = HOG_Detector(name='hog_preprocessed_skimmia')

    hog_pre_skimmia_detector.train(images=preprocessed_skimmia_images_train, 
                                   annotations=skimmia_bboxes_train,
                                   max_neg_samples=20000)
    
    hog_pre_skimmia_detector.evaluate(images=preprocessed_skimmia_images_valid, 
                                      annotations=skimmia_bboxes_valid,
                                      pred_threshold=0.6)
    
    hog_pre_skimmia_pred_bboxes, hog_pre_skimmia_pred_scores, hog_pre_skimmia_avg_pred_time = hog_pre_skimmia_detector.predict(images=skimmia_images_test,
                                                                                                                               threshold=0.6)

    avg_IoU_after_NMS_with_given_tsh(gt_bboxes=skimmia_bboxes_test,
                                     pred_bboxes=hog_pre_skimmia_pred_bboxes,
                                     pred_scores=hog_pre_skimmia_pred_scores,
                                     thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
                                     name='hog_pre_skimmia')

    print(f"Average prediction time: {hog_pre_skimmia_avg_pred_time:.2f} s")

    hog_pre_skimmia_avg_iou, hog_pre_skimmia_selected_bboxes = average_IoU_after_NMS(gt_bboxes=skimmia_bboxes_test,
                                                                                     pred_bboxes=hog_pre_skimmia_pred_bboxes,
                                                                                     pred_scores=hog_pre_skimmia_pred_scores,
                                                                                     nms_threshold=0.1)
                                       
    plot_predictions(images=skimmia_images_test[:5],
                     gt_bboxes=skimmia_bboxes_test[:5],
                     pred_bboxes=hog_pre_skimmia_selected_bboxes[:5],
                     name='hog_pre_skimmia_images_predictions')

   
    #-------------------- TRAINING RCNN MODEL (SKIMMIA PREPROCESSED) --------------------
    print('\n')
    print('-------------------- RCNN --------------------')

    cnn_pre_skimmia = get_skimmia_cnn()

    rcnn_pre_skimmia = RCNN(resize=(96, 96),
                            cnn=cnn_pre_skimmia,
                            name='rcnn_skimmia_preprocessed')

    rcnn_pre_skimmia.train(images=preprocessed_skimmia_images_train,
                           annotations=skimmia_bboxes_train,
                           max_neg_samples=30000,
                           epochs=80)
    
    rcnn_pre_skimmia.evaluate(images=preprocessed_skimmia_images_valid,
                              annotations=skimmia_bboxes_valid,
                              pred_threshold=0.6)

    rcnn_pre_skimmia_pred_rois, rcnn_pre_skimmia_pred_bboxes, rcnn_pre_skimmia_pred_scores, rcnn_pre_skimmia_avg_pred_time = rcnn_pre_skimmia.predict(images=preprocessed_skimmia_images_test, 
                                                                                                                                                      threshold=0.6)

    avg_IoU_after_NMS_with_given_tsh(gt_bboxes=skimmia_bboxes_test,
                                     pred_bboxes=rcnn_pre_skimmia_pred_bboxes,
                                     pred_scores=rcnn_pre_skimmia_pred_scores,
                                     thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
                                     name='rcnn_pre_skimmia')

    print(f"Average prediction time: {rcnn_pre_skimmia_avg_pred_time:.2f} s")

    rcnn_pre_skimmia_avg_iou, rcnn_pre_skimmia_selected_bboxes = average_IoU_after_NMS(gt_bboxes=skimmia_bboxes_test,
                                                                                       pred_bboxes=rcnn_pre_skimmia_pred_bboxes,
                                                                                       pred_scores=rcnn_pre_skimmia_pred_scores,
                                                                                       nms_threshold=0.1)

    plot_predictions(images=skimmia_images_test[:5],
                     gt_bboxes=skimmia_bboxes_test[:5],
                     pred_bboxes=rcnn_pre_skimmia_selected_bboxes[:5],
                     name='rcnn_pre_skimmia_images_predictions')
    
    #-------------------- LOADING VISEM DATASET --------------------
    print('\n')
    print("Loading visem data... \n")

    visem_images_train, visem_bboxes_train = load_data(VISEM_IMAGES_TRAIN, VISEM_ANNOTATIONS_TRAIN)
    print(f"Loaded {visem_images_train.shape[0]} training images from Visem dataset.")

    visem_images_valid, visem_bboxes_valid = load_data(VISEM_IMAGES_VALID, VISEM_ANNOTATIONS_VALID)
    print(f"Loaded {visem_images_valid.shape[0]} validating images from Visem dataset.")

    visem_images_test, visem_bboxes_test = load_data(VISEM_IMAGES_TEST, VISEM_ANNOTATIONS_TEST)
    print(f"Loaded {visem_images_test.shape[0]} testing images from Visem dataset.")
    
    
    #-------------------- TRAINING HOG DETECTOR (VISEM) --------------------
    print('\n')
    print('-------------------- HOG --------------------') 

    hog_visem_detector = HOG_Detector(base_window_size=(16,16), 
                                      window_sizes=((8,8), (16,16), (32,32)),
                                      name='hog_visem')

    hog_visem_detector.train(images=visem_images_train, 
                             annotations=visem_bboxes_train,
                             max_neg_samples=20000)
    
    hog_visem_detector.evaluate(images=visem_images_valid, 
                                annotations=visem_bboxes_valid,
                                pred_threshold=0.6)
    
    hog_visem_pred_bboxes, hog_visem_pred_scores, hog_visem_avg_pred_time = hog_visem_detector.predict(images=visem_images_test,
                                                                                                       threshold=0.6)

    avg_IoU_after_NMS_with_given_tsh(gt_bboxes=visem_bboxes_test,
                                     pred_bboxes=hog_visem_pred_bboxes,
                                     pred_scores=hog_visem_pred_scores,
                                     thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
                                     name='hog_visem')

    print(f"Average prediction time: {hog_visem_avg_pred_time:.2f} s")

    hog_visem_avg_iou, hog_visem_selected_bboxes = average_IoU_after_NMS(gt_bboxes=visem_bboxes_test,
                                                                         pred_bboxes=hog_visem_pred_bboxes,
                                                                         pred_scores=hog_visem_pred_scores,
                                                                         nms_threshold=0.1)
                                       
    plot_predictions(images=visem_images_test[:5],
                     gt_bboxes=visem_bboxes_test[:5],
                     pred_bboxes=hog_visem_selected_bboxes[:5],
                     name='hog_visem_images_predictions')

    
    #-------------------- TRAINING RCNN MODEL (VISEM) --------------------
    print('\n')
    print('-------------------- RCNN --------------------')

    cnn_visem = get_visem_cnn()

    rcnn_visem = RCNN(resize=(48, 48),
                      cnn=cnn_visem,
                      name='rcnn_visem')

    rcnn_visem.train(images=visem_images_train,
                     annotations=visem_bboxes_train,
                     max_neg_samples=30000,
                     epochs=80)
    
    rcnn_visem.evaluate(images=visem_images_valid,
                        annotations=visem_bboxes_valid,
                        pred_threshold=0.6)

    rcnn_visem_pred_rois, rcnn_visem_pred_bboxes, rcnn_visem_pred_scores, rcnn_visem_avg_pred_time = rcnn_visem.predict(images=visem_images_test, 
                                                                                                                        threshold=0.6)

    avg_IoU_after_NMS_with_given_tsh(gt_bboxes=visem_bboxes_test,
                                     pred_bboxes=rcnn_visem_pred_bboxes,
                                     pred_scores=rcnn_visem_pred_scores,
                                     thresholds=[0.1, 0.15, 0.2, 0.25, 0.3],
                                     name='rcnn_visem')

    print(f"Average prediction time: {rcnn_visem_avg_pred_time:.2f} s")

    rcnn_visem_avg_iou, rcnn_visem_selected_bboxes = average_IoU_after_NMS(gt_bboxes=visem_bboxes_test,
                                                                           pred_bboxes=rcnn_visem_pred_bboxes,
                                                                           pred_scores=rcnn_visem_pred_scores,
                                                                           nms_threshold=0.1)

    plot_predictions(images=visem_images_test[:5],
                     gt_bboxes=visem_bboxes_test[:5],
                     pred_bboxes=rcnn_visem_selected_bboxes[:5],
                     name='rcnn_visem_images_predictions')

if __name__ == '__main__':
    main()