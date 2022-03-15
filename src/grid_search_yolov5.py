import os
import datetime
import time

from pathlib import Path

from yolov5_inference import inference_yolov5

project = "podrydchiki/attributes"

input_gt = f"data/yolov5_inference/{project}/input/gt_images_txts"
image_ext = 'jpg'

output_folder = f"data/grid_search/yolov5/{project}"
if not Path(output_folder).exists():
    Path(output_folder).mkdir(parents=True, exist_ok=True)

model_path = f"yolov5/runs/train/exp38/weights/best.pt"
class_names_path = f"data/yolov5_inference/{project}/input/cfg/obj.names"
with open(class_names_path) as file:
    classes_names = file.readlines()
    classes_names = [d.replace("\n", "") for d in classes_names]
classes_inds = list(range(len(classes_names)))

image_size = 256
map_iou = 0.8
map_calc = True
save_output = False

output_annot_dir = ""
output_images_vis_dir = ""

results = {}
results['project'] = project
results['config'] = model_path
results['weight'] = model_path
results['obj'] = "--//--"
results['names'] = class_names_path
results['-1- '] = "-1-"

best_map_values = best_precision_values = best_recall_values = [0, 0, 0, 0, 0]
best_map = 0.0
best_precision = 0.0
best_recall = 0.0

exp_number = 1
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
nmses = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
                                                  model_path,
                                                  image_size,
                                                  classes_names,
                                                  classes_inds,
                                                  threshold=th,
                                                  nms=nms,
                                                  map_calc=map_calc,
                                                  map_iou=map_iou,
                                                  verbose=False,
                                                  save_output=save_output)

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
        print(f"best: mAP: {best_map}, precision: {best_precision}, recall: {best_recall}")

results["-2- "] = "-2-"
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
