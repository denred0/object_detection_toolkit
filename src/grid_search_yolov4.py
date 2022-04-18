import os
import datetime
import time
from typing import List

import torch

from pathlib import Path

from yolov4_inference import inference_yolov4


def grid_search_yolov4(project: str,
                       input_images: str,
                       output_folder: str,
                       output_annot_dir: str,
                       output_images_vis_dir: str,
                       output_crops_path: str,
                       config_path: str,
                       weight_path: str,
                       meta_path: str,
                       class_names_path: str,
                       hier_thresh: float,
                       thresholds: List,
                       nmses: List,
                       images_ext: str,
                       map_iou: float,
                       save_output: bool,
                       set_custom_input_size: bool):
    #
    results = {}
    results['project'] = project
    results['config'] = config_path
    results['weight'] = weight_path
    results['obj'] = meta_path
    results['names'] = class_names_path
    results['-1-'] = "--"

    best_map_values = best_precision_values = best_recall_values = [0, 0, (0, 0), 0, 0, 0]
    best_map = 0.0
    best_precision = 0.0
    best_recall = 0.0

    exp_number = 1

    custom_input_size_wh = [(608, 608)]
    # thresholds = [0.5]
    # nmses = [0.4]

    exp_count = len(thresholds) * len(nmses) * len(custom_input_size_wh)

    for custom_input_size in custom_input_size_wh:
        for th in thresholds:
            for nms in nmses:

                print(f"\nexp: {exp_number} / {exp_count}, threshold={th}, nms={nms}, input_size={custom_input_size}")

                map, precision, recall = inference_yolov4(input_images,
                                                          output_annot_dir,
                                                          output_images_vis_dir,
                                                          output_crops_path,
                                                          config_path,
                                                          set_custom_input_size,
                                                          custom_input_size,
                                                          weight_path,
                                                          meta_path,
                                                          threshold=th,
                                                          hier_thresh=hier_thresh,
                                                          nms_coeff=nms,
                                                          images_ext=images_ext,
                                                          map_calc=True,
                                                          map_iou=map_iou,
                                                          verbose=False,
                                                          save_output=save_output,
                                                          save_crops=False)

                results[
                    f'exp_{exp_number}'] = f"threshold: {th}, nms: {nms}, input_size: {custom_input_size}, mAP: {map}, precision: {precision}, recall: {recall}"
                exp_number += 1

                if map > best_map:
                    best_map_values = [th, nms, custom_input_size, map, precision, recall]
                    best_map = map

                if precision > best_precision:
                    best_precision_values = [th, nms, custom_input_size, map, precision, recall]
                    best_precision = precision

                if recall > best_recall:
                    best_recall_values = [th, nms, custom_input_size, map, precision, recall]
                    best_recall = recall

                print(f"current: mAP: {map}, precision: {precision}, recall: {recall}")
                print(f"best: mAP: {best_map_values[2]}, "
                      f"precision: {best_map_values[3]}, "
                      f"recall: {best_map_values[4]}, "
                      f"th: {best_map_values[0]}, "
                      f"nms: {best_map_values[1]}")

    results["-2-"] = "--"
    results['best_map'] = f"threshold: {best_map_values[0]}, " \
                          f"nms: {best_map_values[1]}, " \
                          f"input_size: {best_map_values[2]} " \
                          f"mAP: {best_map_values[3]}, " \
                          f"precision: {best_map_values[4]}, " \
                          f"recall: {best_map_values[5]}"

    results['best_precision'] = f"threshold: {best_precision_values[0]}, " \
                                f"nms: {best_precision_values[1]}, " \
                                f"input_size: {best_precision_values[2]}, " \
                                f"mAP: {best_precision_values[3]}, " \
                                f"precision: {best_precision_values[4]}, " \
                                f"recall: {best_precision_values[5]}"

    results['best_recall'] = f"threshold: {best_recall_values[0]}, " \
                             f"nms: {best_recall_values[1]}, " \
                             f"input_size: {best_recall_values[2]}, " \
                             f"mAP: {best_recall_values[3]}, " \
                             f"precision: {best_recall_values[4]}, " \
                             f"recall: {best_recall_values[5]}"

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    with open(os.path.join(output_folder, f"grid_search_result_{timestamp}.txt"), 'w') as f:
        for key, value in results.items():
            f.write("%s\n" % (str(key) + ": " + str(value)))


if __name__ == "__main__":
    # project = "podrydchiki/attributes"
    project = "door_smoke"

    input_images = f"data/yolov4_inference/{project}/input/gt_images_txts"

    output_annot_dir = ""
    output_images_vis_dir = ""
    output_crops_path = ""
    output_folder = f"data/grid_search/yolov4/{project}"
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    config_path = f"data/yolov4_inference/{project}/input/cfg/yolov4-obj-mycustom.cfg"
    weight_path = f"data/yolov4_inference/{project}/input/cfg/yolov4-obj-mycustom_best.weights"
    meta_path = f"data/yolov4_inference/{project}/input/cfg/obj.data"
    class_names_path = f"data/yolov4_inference/{project}/input/cfg/obj.names"
    hier_thresh = 0.3
    images_ext = 'jpg'
    map_iou = 0.8
    save_output = False
    set_custom_input_size = False

    thresholds = [0.3, 0.4, 0.5, 0.6]
    nmses = [0.4, 0.5, 0.6]

    grid_search_yolov4(project,
                       input_images,
                       output_folder,
                       output_annot_dir,
                       output_images_vis_dir,
                       output_crops_path,
                       config_path,
                       weight_path,
                       meta_path,
                       class_names_path,
                       hier_thresh,
                       thresholds,
                       nmses,
                       images_ext,
                       map_iou,
                       save_output,
                       set_custom_input_size)
