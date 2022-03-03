from tqdm import tqdm

from yolov5_inference import inference_yolov5
from my_utils import recreate_folder

project = "podrydchiki/attributes"

input_gt = f"data/yolov5_inference/{project}/input/gt_images_txts"
image_ext = 'jpg'

model_path = f"yolov5/runs/train/exp34/weights/best.pt"
class_names_path = f"data/yolov5_inference/{project}/input/cfg/obj.names"
classes_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# classes_inds = [0]
image_size = 320
map_iou = 0.8
map_calc = True
save_output = False

output_annot_dir = ""
output_images_vis_dir = ""

results = {}
results['project'] = project
results['config'] = model_path
results['weight'] = model_path
results['obj'] = ""
results['names'] = class_names_path
results['-- '] = "--"

best_map = 0.0
best_map_values = [0, 0, 0, 0, 0]
best_precision = 0.0
best_precision_values = [0, 0, 0, 0, 0]
best_recall = 0.0
best_recall_values = [0, 0, 0, 0, 0]

exp_number = 0
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
nmses = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
                                                  class_names_path,
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

        if precision > best_precision:
            best_precision_values = [th, nms, map, precision, recall]

        if recall > best_recall:
            best_recall_values = [th, nms, map, precision, recall]

results["-- "] = "--"
results[
    'best_map'] = f"threshold: {best_map_values[0]}, nms: {best_map_values[1]}, mAP: {best_map_values[2]}, precision: {best_map_values[3]}, recall: {best_map_values[4]}"

results[
    'best_precision'] = f"threshold: {best_precision_values[0]}, nms: {best_precision_values[1]}, mAP: {best_precision_values[2]}, precision: {best_precision_values[3]}, recall: {best_precision_values[4]}"

results[
    'best_recall'] = f"threshold: {best_recall_values[0]}, nms: {best_recall_values[1]}, mAP: {best_recall_values[2]}, precision: {best_recall_values[3]}, recall: {best_recall_values[4]}"

with open(f'data/grid_search/yolov5/{project}.txt', 'w') as f:
    for key, value in results.items():
        f.write("%s\n" % (str(key) + ": " + str(value)))
