import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from processing import NMS
import json

def IoU(box1, box2):
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

def average_IoU(gt_bboxes, pred_bboxes):
    """
    Compute average IoU across given ground truth bboxes and predicted bboxes for images.
    """
    num_images = gt_bboxes.shape[0]
    max_ious = []
    for i in range(num_images):
        gt_bboxes_per_img = gt_bboxes[i]
        pred_bboxes_per_img = pred_bboxes[i]

        for pred_bbox in pred_bboxes_per_img:
            ious = [IoU(box1=gt_box, box2=pred_bbox) for gt_box in gt_bboxes_per_img]
            if len(ious) != 0:
                max_ious.append(np.max(ious))
            #else:
            #    max_ious.append(0)

    return np.average(max_ious)

def average_IoU_after_NMS(gt_bboxes, pred_bboxes, pred_scores, nms_threshold):
    """
    Compute average IoU for compressed bounding boxes.
    """
    selected_bboxes = []

    for img_bboxes, img_scores in zip(pred_bboxes, pred_scores):
        if len(img_bboxes) != 0:
            selected_bboxes.append(NMS(bboxes=img_bboxes,
                                       scores=img_scores,
                                       iou_threshold=nms_threshold))
        else:
            selected_bboxes.append([])

    avg_iou = average_IoU(gt_bboxes=gt_bboxes,
                          pred_bboxes=selected_bboxes)
        
    #print(f"Average IoU after NMS with {nms_threshold} threshold is equals {avg_iou:.2f}")

    return avg_iou, np.array(selected_bboxes)

def avg_IoU_after_NMS_with_given_tsh(gt_bboxes,
                                     pred_bboxes,
                                     pred_scores,
                                     thresholds, 
                                     name):
    results = {}
    for tsh in thresholds:
        avg_iou = average_IoU_after_NMS(gt_bboxes=gt_bboxes,
                                        pred_bboxes=pred_bboxes,
                                        pred_scores=pred_scores,
                                        nms_threshold=tsh)[0]
        
        results[f'{tsh}'] = round(avg_iou, 2)
    
    print(results)

    with open(f"results/{name}.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved as resuts/{name}.json")

def plot_predictions(images, gt_bboxes, pred_bboxes, name):
    """
    Plot given bboxes on given images and save in plots directory.
    """
    fig, ax = plt.subplots(1, len(images), figsize=(20, 10))
    
    for i, image in enumerate(images):
        
        for gt_box in gt_bboxes[i]:
            x_center, y_center, width, height = gt_box
            top_left = (int(x_center - width/2), int(y_center - height/2))
            bottom_right = (int(x_center + width/2), int(y_center + height/2))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        for pred_box in pred_bboxes[i]:
            x_center, y_center, width, height = pred_box
            top_left = (int(x_center - width/2), int(y_center - height/2))
            bottom_right = (int(x_center + width/2), int(y_center + height/2))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

        ax[i].imshow(image)
        ax[i].axis('off')

    save_path = os.path.join('plots/', str(name + '.png'))
    plt.savefig(save_path)