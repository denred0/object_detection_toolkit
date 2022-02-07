from tqdm import tqdm

from yolov4_inference import inference
from utils import recreate_folder

project = "barrier_reef"

input_images = f"data/yolo4_inference/{project}/input/images"
input_annot = f"data/yolo4_inference/{project}/input/annot_gt"

config_path = f"data/yolo4_inference/{project}/input/cfg/yolov4-obj-mycustom.cfg"
weight_path = f"data/yolo4_inference/{project}/input/cfg/yolov4-obj-mycustom_best.weights"
meta_path = f"data/yolo4_inference/{project}/input/cfg/obj.data"
class_names_path = f"data/yolo4_inference/{project}/input/cfg/obj.names"
threshold = 0.2
hier_thresh = 0.3
nms_coeff = 0.5
images_ext = 'png'

output_annot_dir = f"data/yolo4_inference/{project}/output/annot_pred"
recreate_folder(output_annot_dir)
output_images_vis_dir = f"data/yolo4_inference/{project}/output/images_vis"
recreate_folder(output_images_vis_dir)

results = {}
results['project'] = project
results['config'] = config_path
results['weight'] = weight_path
results['obj'] = meta_path
results['names'] = class_names_path
results['-'] = "-"

best_map = 0.0
best_map_values = [0, 0, 0, 0, 0]
best_precision = 0.0
best_precision_values = [0, 0, 0, 0, 0]
best_recall = 0.0
best_recall_values = [0, 0, 0, 0, 0]

exp_number = 0
thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
nmses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# thresholds = [0.5]
# nmses = [0.4]

for th in thresholds:
    for nms in nmses:

        print(f"exp: {exp_number}, threshold={th}, nms={nms}")

        map, precision, recall = inference(input_images,
                                           input_annot,
                                           output_annot_dir,
                                           output_images_vis_dir,
                                           config_path, weight_path,
                                           meta_path,
                                           class_names_path,
                                           threshold=th,
                                           hier_thresh=hier_thresh,
                                           nms_coeff=nms,
                                           images_ext=images_ext,
                                           map_calc=True,
                                           verbose=False)

        results[
            f'exp_{exp_number}'] = f"threshold: {th}, nms: {nms}, mAP: {map}, precision: {precision}, recall: {recall}"
        exp_number += 1

        if map > best_map:
            best_map_values = [th, nms, map, precision, recall]

        if precision > best_precision:
            best_precision_values = [th, nms, map, precision, recall]

        if recall > best_recall:
            best_recall_values = [th, nms, map, precision, recall]

results["--"] = "--"
results[
    'best_map'] = f"threshold: {best_map_values[0]}, nms: {best_map_values[1]}, mAP: {best_map_values[2]}, precision: {best_map_values[3]}, recall: {best_map_values[4]}"

results[
    'best_precision'] = f"threshold: {best_precision_values[0]}, nms: {best_precision_values[1]}, mAP: {best_precision_values[2]}, precision: {best_precision_values[3]}, recall: {best_precision_values[4]}"

results[
    'best_recall'] = f"threshold: {best_recall_values[0]}, nms: {best_recall_values[1]}, mAP: {best_recall_values[2]}, precision: {best_recall_values[3]}, recall: {best_recall_values[4]}"

with open(f'data/grid_search/{project}.txt', 'w') as f:
    for key, value in results.items():
        f.write("%s\n" % (str(key) + ": " + str(value)))
