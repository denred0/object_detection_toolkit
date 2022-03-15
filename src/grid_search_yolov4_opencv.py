import os
import datetime
import time

from pathlib import Path

from yolov4_inference_opencv import inference_yolov4_opencv


def grid_search_yolov4_opencv(project,
                              input_images,
                              output_folder,
                              output_annot_dir,
                              output_images_vis_dir,
                              config_path,
                              meta_path,
                              weight_path,
                              class_names_path,
                              class_names,
                              images_ext,
                              map_iou,
                              save_output):
    #

    best_map_values = best_precision_values = best_recall_values = [0, 0, (0, 0), 0, 0, 0]
    best_map = 0.0
    best_precision = 0.0
    best_recall = 0.0

    exp_number = 1
    thresholds = [0.4, 0.5]
    nmses = [0.5, 0.6]
    custom_input_size_wh = [(416, 416)]
    # thresholds = [0.5]
    # nmses = [0.4]
    exp_count = len(thresholds) * len(nmses) * len(custom_input_size_wh)

    results = {}
    results['project'] = project
    results['config'] = config_path
    results['weight'] = weight_path
    results['obj'] = meta_path
    results['names'] = class_names_path
    results['--'] = "--"

    for custom_input_size in custom_input_size_wh:
        for th in thresholds:
            for nms in nmses:

                print(f"\nexp: {exp_number} / {exp_count}, threshold={th}, nms={nms}, input_size={custom_input_size}")

                map, precision, recall = inference_yolov4_opencv(input_images,
                                                                 output_annot_dir,
                                                                 output_images_vis_dir,
                                                                 config_path,
                                                                 class_names,
                                                                 custom_input_size,
                                                                 weight_path,
                                                                 threshold=th,
                                                                 nms_coeff=nms,
                                                                 images_ext=images_ext,
                                                                 map_calc=True,
                                                                 map_iou=map_iou,
                                                                 verbose=False,
                                                                 save_output=save_output)

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
                print(f"best: mAP: {best_map}, precision: {best_precision}, recall: {best_recall}")

    results["--"] = "--"
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
    project = "podrydchiki/persons"

    input_images = f"data/yolov4_inference_opencv/{project}/input/gt_images_txts"

    output_folder = f"data/grid_search/yolov4_opencv/{project}"
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    class_names_path = f"data/yolov4_inference_opencv/{project}/input/cfg/obj.names"
    with open(class_names_path) as file:
        class_names = file.readlines()
        class_names = [d.replace("\n", "") for d in class_names]

    config_path = f"data/yolov4_inference_opencv/{project}/input/cfg/yolov4-obj-mycustom.cfg"
    weight_path = f"data/yolov4_inference_opencv/{project}/input/cfg/yolov4-obj-mycustom_best.weights"
    meta_path = f"data/yolov4_inference/{project}/input/cfg/obj.data"
    images_ext = 'jpg'
    map_iou = 0.8
    save_output = False

    output_annot_dir = ""
    output_images_vis_dir = ""

    grid_search_yolov4_opencv(project,
                              input_images,
                              output_folder,
                              output_annot_dir,
                              output_images_vis_dir,
                              config_path,
                              meta_path,
                              weight_path,
                              class_names_path,
                              class_names,
                              images_ext,
                              map_iou,
                              save_output)
