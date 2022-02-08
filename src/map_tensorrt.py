import cv2
import os
import numpy as np

from tqdm import tqdm

from my_utils import get_all_files_in_folder
from map import mean_average_precision

gt_input_dir = "data/compare_torch_tensorrt/input/gt_images_txts"
pred_input_dir = "data/compare_torch_tensorrt/input/pred_txts/YOLOV5_FP16"
image_ext = "jpg"

num_classes = 1
iou_threshold = 0.5

gt_txts = get_all_files_in_folder(gt_input_dir, ["*.txt"])

map_images = []
precision_images = []
recall_images = []

for g_txt in tqdm(gt_txts):
    img = cv2.imread(os.path.join(gt_input_dir, g_txt.stem + "." + image_ext), cv2.IMREAD_COLOR)

    h, w = img.shape[:2]

    with open(g_txt) as file:
        detections_gt = file.readlines()
        detections_gt = [d.replace("\n", "") for d in detections_gt]
        detections_gt = [d.split() for d in detections_gt]
        detections_gt = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])] for d in detections_gt]

    detections_gt_abs = []
    for det in detections_gt:
        xmin = int(det[1] * w - (det[3] * w / 2))
        xmax = int(det[1] * w + (det[3] * w / 2))

        ymin = int(det[2] * h - (det[4] * h / 2))
        ymax = int(det[2] * h + (det[4] * h / 2))

        detections_gt_abs.append([det[0], xmin, ymin, xmax, ymax])

    with open(os.path.join(pred_input_dir, g_txt.name)) as file:
        detections_result = file.readlines()
        detections_result = [d.replace("\n", "") for d in detections_result]
        detections_result = [d.split() for d in detections_result]
        detections_result = [
            [int(d[0]), int(float(d[1])), int(float(d[2])), int(float(d[3])), int(float(d[4])), int(float(d[5]))] for d
            in
            detections_result]

    if detections_result == detections_gt_abs == []:
        map_images.append(1)
        precision_images.append(1)
        recall_images.append(1)
    elif detections_gt_abs == [] and detections_result != []:
        map_images.append(0)
        precision_images.append(0)
        recall_images.append(0)
    else:
        map_image, precision_image, recall_image = mean_average_precision(pred_boxes=detections_result,
                                                                          true_boxes=detections_gt_abs,
                                                                          num_classes=num_classes,
                                                                          iou_threshold=iou_threshold)
        map_images.append(map_image)
        precision_images.append(precision_image)
        recall_images.append(recall_image)

print(f"Images count: {len(gt_txts)}")
print(f"mAP: {round(np.mean(map_images), 4)}")

# precision - не находим лишнее (уменьшаем FP)
print(f"Precision: {round(np.mean(precision_images), 4)}")

# recall - находим все объекты (уменьшаем FN)
print(f"Recall: {round(np.mean(recall_images), 4)}")
