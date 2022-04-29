import time
import numpy as np
import cv2
import os
import torch

from pathlib import Path
from typing import List
from tqdm import tqdm
from collections import defaultdict

from my_utils import recreate_folder, get_all_files_in_folder, plot_one_box
from map import mean_average_precision


def inference_yolov5(input_gt: str,
                     image_ext: str,
                     output_annot_dir: str,
                     output_images_vis_dir: str,
                     output_crops_path: str,
                     model_path: str,
                     image_size: int,
                     classes_names: List,
                     classes_inds: List,
                     threshold=0.5,
                     nms=0.5,
                     map_calc=True,
                     map_iou=0.5,
                     verbose=True,
                     save_output=True,
                     draw_gt=True,
                     draw_predictions=True,
                     save_crops=True) -> [float, float, float]:
    #
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.conf = threshold
    model.iou = nms
    model.classes = classes_inds

    images = get_all_files_in_folder(input_gt, [f"*.{image_ext}"])

    map_images = []
    map_classes_total = defaultdict(list)
    precision_images = []
    recall_images = []

    detection_time = 0

    for im in tqdm(images):

        img = cv2.imread(str(im))
        img_orig = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        start = time.time()
        results = model(img, size=image_size)
        detection_time += time.time() - start
        results_list = results.pandas().xyxy[0].values.tolist()

        detections_result = []

        detections_valid = [d for d in results_list if float(d[4]) > threshold]

        for ind, res in enumerate(detections_valid):
            (xmin, ymin) = (res[0], res[1])
            (xmax, ymax) = (res[2], res[3])
            width = xmax - xmin
            height = ymax - ymin

            x_center_norm = float((xmax - xmin) / 2 + xmin) / w
            y_center_norm = float((ymax - ymin) / 2 + ymin) / h
            w_norm = float(width) / w
            h_norm = float(height) / h

            if w_norm > 1: w_norm = 1.0
            if h_norm > 1: h_norm = 1.0

            detections_result.append(
                [
                    res[5],
                    round(res[4], 2),
                    x_center_norm,
                    y_center_norm,
                    w_norm,
                    h_norm
                ])

            if save_crops:
                increase_coef = 0.2
                width_crop = xmax - xmin
                height_crop = ymax - ymin
                xmin_ = int(xmin - width_crop * increase_coef) if int(xmin - width_crop * increase_coef) > 0 else 0
                xmax_ = int(xmax + width_crop * increase_coef) if int(xmax + width_crop * increase_coef) < w else w
                ymin_ = int(ymin - height_crop * increase_coef) if int(ymin - height_crop * increase_coef) > 0 else 0
                ymax_ = int(ymax + height_crop * increase_coef) if int(ymax + height_crop * increase_coef) < h else h

                person_crop_orig = img_orig[ymin_:ymax_, xmin_:xmax_, :]
                cv2.imwrite(os.path.join(output_crops_path, f"{im.stem}_{ind}.{image_ext}"), person_crop_orig)

            if draw_predictions:
                img_orig = plot_one_box(img_orig, [int(xmin), int(ymin), int(xmax), int(ymax)],
                                        str(res[6] + " " + str(round(res[4], 2))),
                                        color=(255, 255, 0), write_label=False)

        if map_calc:
            with open(Path(input_gt).joinpath(im.stem + ".txt")) as file:
                detections_gt = file.readlines()
                detections_gt = [d.replace("\n", "") for d in detections_gt]
                detections_gt = [d.split() for d in detections_gt]
                detections_gt = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])] for d in detections_gt]

            if detections_result == detections_gt == []:
                map_images.append(1)
                precision_images.append(1)
                recall_images.append(1)
            elif detections_gt == [] and detections_result != []:
                map_images.append(0)
                precision_images.append(0)
                recall_images.append(0)
            else:
                map_image, precision_image, recall_image, map_classes = mean_average_precision(
                    pred_boxes=detections_result,
                    true_boxes=detections_gt,
                    num_classes=len(classes_names),
                    iou_threshold=map_iou)
                map_images.append(map_image)
                precision_images.append(precision_image)
                recall_images.append(recall_image)

                for cl, ap in map_classes.items():
                    map_classes_total[cl].append(ap)

            if draw_gt:
                for det_gt in detections_gt:
                    x_top = int((det_gt[1] - det_gt[3] / 2) * w)
                    y_top = int((det_gt[2] - det_gt[4] / 2) * h)
                    x_bottom = int((x_top + det_gt[3] * w))
                    y_bottom = int((y_top + det_gt[4] * h))

                    class_gt = classes_names[det_gt[0]]

                    img_orig = plot_one_box(img_orig, [int(x_top), int(y_top), int(x_bottom), int(y_bottom)],
                                            str(class_gt),
                                            color=(0, 255, 0))

        if save_output:
            with open(Path(output_annot_dir).joinpath(im.stem + '.txt'), 'w') as f:
                for item in detections_result:
                    f.write("%s\n" % (str(item[0]) + ' ' + str(item[2]) + ' ' + str(item[3]) + ' ' + str(
                        item[4]) + ' ' + str(item[5])))

            cv2.imwrite(str(Path(output_images_vis_dir).joinpath(im.name)), img_orig)

    if verbose:
        print(f"Images count: {len(images)}")
        print(f"mAP: {round(np.mean(map_images), 4)}")
        # precision - не находим лишнее (уменьшаем FP)
        print(f"Precision: {round(np.mean(precision_images), 4)}")
        # recall - находим все объекты (уменьшаем FN)
        print(f"Recall: {round(np.mean(recall_images), 4)}")
        print(f"th: {threshold}")
        print(f"nms: {nms}")

        print()
        for key, value in map_classes_total.items():
            print(f"{classes_names[key]}: {round(sum(value) / len(value), 4)}")

        print()
        print(f"mAP IoU: {map_iou}")
        print(f"FPS: {round(len(images) / detection_time, 2)}")

    return round(np.mean(map_images), 4), round(np.mean(precision_images), 4), round(np.mean(recall_images), 4)


if __name__ == '__main__':
    # project = "evraz/attr"
    # project = "evraz/persons"
    # project = "podrydchiki/persons"
    # project = "podrydchiki/attributes"
    # project = "gaz_door"
    # project = "rosatom/persons"
    project = "rosatom/attr"

    input_gt = f"data/yolov5_inference/{project}/input/gt_images_txts"
    image_ext = "jpg"

    # model_path = f"yolov5/runs/train/podrydchiki/person/exp/weights/best.pt"
    # model_path = f"yolov5/runs/train/exp8/weights/last.pt"
    # model_path = f"data/yolov5_inference/podrydchiki/attributes/input/cfg/best.pt"
    # model_path = f"yolov5/runs/train/podrydchiki/attr/exp3/weights/best.pt"
    model_path = f"data/yolov5_inference/{project}/input/cfg/best.pt"
    class_names_path = f"data/yolov5_inference/{project}/input/cfg/obj.names"
    with open(class_names_path) as file:
        classes_names = file.readlines()
        classes_names = [d.replace("\n", "") for d in classes_names]
    classes_inds = list(range(len(classes_names)))

    threshold = 0.7
    nms = 0.4
    image_size = 320
    # image_size = 640
    map_iou = 0.8
    map_calc = True
    save_output = True
    draw_gt = False
    draw_predictions = True
    save_crops = False

    output_annot_dir = f"data/yolov5_inference/{project}/output/annot_pred"
    recreate_folder(output_annot_dir)
    output_images_vis_dir = f"data/yolov5_inference/{project}/output/images_vis"
    recreate_folder(output_images_vis_dir)

    output_crops_path = f"data/yolov5_inference/{project}/output/crops"
    if save_crops:
        recreate_folder(output_crops_path)

    inference_yolov5(input_gt,
                     image_ext,
                     output_annot_dir,
                     output_images_vis_dir,
                     output_crops_path,
                     model_path,
                     image_size,
                     classes_names,
                     classes_inds,
                     threshold,
                     nms,
                     map_calc=map_calc,
                     map_iou=map_iou,
                     verbose=True,
                     save_output=save_output,
                     draw_gt=draw_gt,
                     draw_predictions=draw_predictions,
                     save_crops=save_crops)
