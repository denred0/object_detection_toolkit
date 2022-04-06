import os
import sys
import cv2
import time
import numpy as np

from tqdm import tqdm
from pathlib import Path
from my_utils import get_all_files_in_folder, recreate_folder, plot_one_box
from collections import defaultdict

from my_darknet import load_network, detect_image
from map import mean_average_precision


def inference_yolov4(input_gt: str,
                     output_annot_dir: str,
                     output_images_vis_dir: str,
                     output_crops_path: str,
                     config_path: str,
                     set_custom_input_size: bool,
                     custom_input_size_wh: tuple,
                     weight_path: str,
                     meta_path: str,
                     threshold=0.5,
                     hier_thresh=0.45,
                     nms_coeff=0.5,
                     images_ext='jpg',
                     map_calc=False,
                     map_iou=0.5,
                     verbose=False,
                     save_output=True,
                     draw_gt=True,
                     draw_predictions=True,
                     save_crops=True) -> [float, float, float]:
    #
    if set_custom_input_size:
        with open(config_path) as file:
            cfg_lines = [line.rstrip() for line in file]

        for id, line in enumerate(cfg_lines):
            if "width" in line:
                cfg_lines[id] = f"width={custom_input_size_wh[0]}"

            if "height" in line:
                cfg_lines[id] = f"height={custom_input_size_wh[1]}"

        filepath = os.sep.join(config_path.split(os.sep)[:-1])
        filename = config_path.split(os.sep)[-1].split(".")[0]

        custom_config_path = os.path.join(filepath, filename + f"_input_size_{custom_input_size_wh}.cfg")
        with open(custom_config_path, 'w') as f:
            for item in cfg_lines:
                f.write("%s\n" % item)

        config_path = custom_config_path
        print(f"Inference with custom input path {custom_input_size_wh}")

    net_main, class_names, colors = load_network(config_path, meta_path, weight_path)

    images = get_all_files_in_folder(input_gt, [f"*.{images_ext}"])

    map_images = []
    map_classes_total = defaultdict(list)
    precision_images = []
    recall_images = []

    detection_time = 0

    for im in tqdm(images):

        img = cv2.imread(str(im), cv2.IMREAD_COLOR)
        img_orig = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        start = time.time()
        detections = detect_image(net_main,
                                  class_names,
                                  img,
                                  thresh=threshold,
                                  hier_thresh=hier_thresh,
                                  nms=nms_coeff)
        detection_time += time.time() - start

        detections_result = []
        detections_valid = [d for d in detections if float(d[1]) / 100 > threshold]

        for ind, detection in enumerate(detections_valid):

            current_class = detection[0]
            # if current_class == "smoke":
            #     print(im.name)

            current_thresh = float(detection[1])
            current_coords = [float(x) for x in detection[2]]

            xmin = float(current_coords[0] - current_coords[2] / 2)
            ymin = float(current_coords[1] - current_coords[3] / 2)
            xmax = float(xmin + current_coords[2])
            ymax = float(ymin + current_coords[3])

            xmin = 0 if xmin < 0 else xmin
            xmax = w if xmax > w else xmax

            ymin = 0 if ymin < 0 else ymin
            ymax = h if ymax > h else ymax

            x_center_norm = float(current_coords[0]) / w
            y_center_norm = float(current_coords[1]) / h
            w_norm = float(current_coords[2]) / w
            h_norm = float(current_coords[3]) / h

            if w_norm > 1: w_norm = 1.0
            if h_norm > 1: h_norm = 1.0

            detections_result.append(
                [
                    class_names.index(current_class),
                    round(current_thresh / 100, 2),
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
                cv2.imwrite(os.path.join(output_crops_path, f"{im.stem}_{ind}.{images_ext}"), person_crop_orig)

            if draw_predictions:
                img_orig = plot_one_box(img_orig, [int(xmin), int(ymin), int(xmax), int(ymax)],
                                        str(current_class + " " + str(round(current_thresh / 100, 2))),
                                        color=(255, 255, 0))

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
                    num_classes=len(class_names),
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

                    class_gt = class_names[det_gt[0]]

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

        print()
        for key, value in map_classes_total.items():
            print(f"{class_names[key]}: {round(sum(value) / len(value), 4)}")

        print()
        print(f"mAP IoU: {map_iou}")
        print(f"FPS: {round(len(images) / detection_time, 2)}")

    return round(np.mean(map_images), 4), round(np.mean(precision_images), 4), round(np.mean(recall_images), 4)


if __name__ == '__main__':
    # project = "evraz/persons"
    project = "podrydchiki/persons"
    # project = "door_smoke"

    input_gt = f"data/yolov4_inference/{project}/input/gt_images_txts"
    images_ext = 'jpg'

    config_path = f"data/yolov4_inference/{project}/input/cfg/yolov4-obj-mycustom.cfg"
    weight_path = f"data/yolov4_inference/{project}/input/cfg/yolov4-obj-mycustom_best.weights"
    meta_path = f"data/yolov4_inference/{project}/input/cfg/obj.data"

    threshold = 0.5
    hier_thresh = 0.3
    nms_coeff = 0.5
    map_iou = 0.8
    map_calc = False
    save_output = True
    draw_gt = False
    save_crops = True
    draw_predictions = False

    set_custom_input_size = False
    custom_input_size_wh = (416, 416)

    output_annot_dir = f"data/yolov4_inference/{project}/output/annot_pred"
    recreate_folder(output_annot_dir)
    output_images_vis_dir = f"data/yolov4_inference/{project}/output/images_vis"
    recreate_folder(output_images_vis_dir)

    output_crops_path = f"data/yolov4_inference/{project}/output/crops"
    if save_crops:
        recreate_folder(output_crops_path)

    inference_yolov4(input_gt,
                     output_annot_dir,
                     output_images_vis_dir,
                     output_crops_path,
                     config_path,
                     set_custom_input_size,
                     custom_input_size_wh,
                     weight_path,
                     meta_path,
                     threshold,
                     hier_thresh,
                     nms_coeff,
                     images_ext,
                     map_calc=map_calc,
                     map_iou=map_iou,
                     verbose=True,
                     save_output=save_output,
                     draw_gt=draw_gt,
                     draw_predictions=draw_predictions,
                     save_crops=save_crops)
