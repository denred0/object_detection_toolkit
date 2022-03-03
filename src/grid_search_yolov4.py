import os
import datetime
import time

from pathlib import Path

from yolov4_inference import inference_yolov4
from my_utils import recreate_folder

project = "barrier_reef"

input_images = f"data/yolov4_inference/{project}/input/gt_images_txts"

output_folder = f"data/grid_search/yolov4/{project}"
if not Path(output_folder).exists():
    Path(output_folder).mkdir(parents=True, exist_ok=True)

config_path = f"data/yolov4_inference/{project}/input/cfg/yolov4-obj-mycustom.cfg"
weight_path = f"data/yolov4_inference/{project}/input/cfg/yolov4-obj-mycustom_best.weights"
meta_path = f"data/yolov4_inference/{project}/input/cfg/obj.data"
class_names_path = f"data/yolov4_inference/{project}/input/cfg/obj.names"
threshold = 0.2
hier_thresh = 0.3
nms_coeff = 0.5
images_ext = 'png'
map_iou = 0.8
save_output = False

output_annot_dir = ""
output_images_vis_dir = ""

results = {}
results['project'] = project
results['config'] = config_path
results['weight'] = weight_path
results['obj'] = meta_path
results['names'] = class_names_path
results['--'] = "--"

best_map = 0.0
best_map_values = [0, 0, 0, 0, 0]
best_precision = 0.0
best_precision_values = [0, 0, 0, 0, 0]
best_recall = 0.0
best_recall_values = [0, 0, 0, 0, 0]

exp_number = 1
thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
nmses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
exp_count = len(thresholds) * len(nmses)
# thresholds = [0.5]
# nmses = [0.4]

for th in thresholds:
    for nms in nmses:

        print(f"\nexp: {exp_number} / {exp_count}, threshold={th}, nms={nms}")

        map, precision, recall = inference_yolov4(input_images,
                                                  output_annot_dir,
                                                  output_images_vis_dir,
                                                  config_path,
                                                  weight_path,
                                                  meta_path,
                                                  class_names_path,
                                                  threshold=th,
                                                  hier_thresh=hier_thresh,
                                                  nms_coeff=nms,
                                                  images_ext=images_ext,
                                                  map_calc=True,
                                                  map_iou=map_iou,
                                                  verbose=False,
                                                  save_output=save_output)

        results[
            f'exp_{exp_number}'] = f"threshold: {th}, nms: {nms}, mAP: {map}, precision: {precision}, recall: {recall}"
        print(f"mAP: {map}, precision: {precision}, recall: {recall}")
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

results["--"] = "--"
results[
    'best_map'] = f"threshold: {best_map_values[0]}, nms: {best_map_values[1]}, mAP: {best_map_values[2]}, precision: {best_map_values[3]}, recall: {best_map_values[4]}"

results[
    'best_precision'] = f"threshold: {best_precision_values[0]}, nms: {best_precision_values[1]}, mAP: {best_precision_values[2]}, precision: {best_precision_values[3]}, recall: {best_precision_values[4]}"

results[
    'best_recall'] = f"threshold: {best_recall_values[0]}, nms: {best_recall_values[1]}, mAP: {best_recall_values[2]}, precision: {best_recall_values[3]}, recall: {best_recall_values[4]}"

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
with open(os.path.join(output_folder, f"grid_search_result_{timestamp}.txt"), 'w') as f:
    for key, value in results.items():
        f.write("%s\n" % (str(key) + ": " + str(value)))
