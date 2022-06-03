import os
import datetime
import time

from pathlib import Path
from typing import List

from yolov5_inference import inference_yolov5


def grid_search_yolov5(project: str,
                       input_gt: str,
                       image_ext: str,
                       output_folder: str,
                       output_annot_dir: str,
                       output_images_vis_dir: str,
                       output_crops_path: str,
                       weight_path: str,
                       thresholds: List,
                       nmses: List,
                       class_names_path: str,
                       classes_names: List,
                       classes_inds: List,
                       image_size: int,
                       map_iou: float,
                       map_calc: bool,
                       save_output: bool,
                       save_crops: bool) -> None:
    #
    results = {}
    results['project'] = project
    results['config'] = weight_path
    results['weight'] = weight_path
    results['obj'] = "--//--"
    results['names'] = class_names_path
    results['-1- '] = "-1-"

    best_map_values = best_precision_values = best_recall_values = [0, 0, 0, 0, 0]
    best_map = 0.0
    best_precision = 0.0
    best_recall = 0.0

    exp_number = 1

    # thresholds = [0.3]
    # nmses = [0.3]
    exp_count = len(thresholds) * len(nmses)

    for th in thresholds:
        for nms in nmses:

            print(f"\nexp: {exp_number} / {exp_count}, threshold={th}, nms={nms}")

            map, precision, recall = inference_yolov5(input_gt,
                                                      image_ext,
                                                      output_annot_dir,
                                                      output_images_vis_dir,
                                                      output_crops_path,
                                                      weight_path,
                                                      image_size,
                                                      classes_names,
                                                      classes_inds,
                                                      threshold=th,
                                                      nms=nms,
                                                      map_calc=map_calc,
                                                      map_iou=map_iou,
                                                      verbose=False,
                                                      save_output=save_output,
                                                      save_crops=save_crops)

            results[
                f'exp_{exp_number}'] = f"threshold: {th}, nms: {nms}, mAP: {map}, precision: {precision}, recall: {recall}"

            exp_number += 1

            if map > best_map:
                best_map_values = [th, nms, map, precision, recall]
                best_map = map

            if precision > best_precision:
                best_precision_values = [th, nms, map, precision, recall]
                best_precision = precision

            if recall > best_recall:
                best_recall_values = [th, nms, map, precision, recall]
                best_recall = recall

            print(f"current: mAP: {map}, precision: {precision}, recall: {recall}")
            print(f"best: mAP: {best_map_values[2]}, "
                  f"precision: {best_map_values[3]}, "
                  f"recall: {best_map_values[4]}, "
                  f"th: {best_map_values[0]}, "
                  f"nms: {best_map_values[1]}")

    results["-2- "] = "-2-"
    results['best_map'] = f"threshold: {best_map_values[0]}, " \
                          f"nms: {best_map_values[1]}, " \
                          f"mAP: {best_map_values[2]}, " \
                          f"precision: {best_map_values[3]}, " \
                          f"recall: {best_map_values[4]}"

    results['best_precision'] = f"threshold: {best_precision_values[0]}, " \
                                f"nms: {best_precision_values[1]}, " \
                                f"mAP: {best_precision_values[2]}, " \
                                f"precision: {best_precision_values[3]}, " \
                                f"recall: {best_precision_values[4]}"

    results['best_recall'] = f"threshold: {best_recall_values[0]}, " \
                             f"nms: {best_recall_values[1]}, " \
                             f"mAP: {best_recall_values[2]}, " \
                             f"precision: {best_recall_values[3]}, " \
                             f"recall: {best_recall_values[4]}"

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    with open(os.path.join(output_folder, f"grid_search_result_{timestamp}.txt"), 'w') as f:
        for key, value in results.items():
            f.write("%s\n" % (str(key) + ": " + str(value)))


if __name__ == "__main__":
    # project = "podrydchiki/attributes"
    # project = "evraz/attr"
    project = "rosatom/attr"

    input_gt = f"data/yolov5_inference/{project}/input/gt_images_txts"
    image_ext = 'jpg'

    output_annot_dir = ""
    output_images_vis_dir = ""
    output_crops_path = ""
    output_folder = f"data/grid_search/yolov5/{project}"
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    weight_path = f"data/yolov5_inference/{project}/input/cfg/best.pt"
    class_names_path = f"data/yolov5_inference/{project}/input/cfg/obj.names"
    with open(class_names_path) as file:
        classes_names = file.readlines()
        classes_names = [d.replace("\n", "") for d in classes_names]
    classes_inds = list(range(len(classes_names)))

    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    nmses = [0.3, 0.4, 0.5, 0.6]
    image_size = 320
    map_iou = 0.8
    map_calc = True
    save_output = False
    save_crops = False

    grid_search_yolov5(project,
                       input_gt,
                       image_ext,
                       output_folder,
                       output_annot_dir,
                       output_images_vis_dir,
                       output_crops_path,
                       weight_path,
                       thresholds,
                       nmses,
                       class_names_path,
                       classes_names,
                       classes_inds,
                       image_size,
                       map_iou,
                       map_calc,
                       save_output,
                       save_crops)
