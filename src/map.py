# source https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/mean_avg_precision.py
import numpy as np
from iou import intersection_over_union_box


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    average_precisions_classes = {}

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[0] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[0] == c:
                ground_truths.append(true_box)

        amount_bboxes = np.zeros(len(ground_truths))

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[1], reverse=True)
        TP = np.zeros((len(detections)))
        FP = np.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            best_iou = 0

            for idx, gt in enumerate(ground_truths):
                iou = intersection_over_union_box(
                    detection[2:],
                    gt[1:],
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = np.cumsum(TP, axis=0)
        FP_cumsum = np.cumsum(FP, axis=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = np.concatenate((np.ones([1]), precisions))
        precisions[1:-1] = precisions[2:]
        recalls = np.concatenate((np.zeros([1]), recalls))
        # np.trapz for numerical integration
        # находит площадь под треугольниками, не достраивая до прямоугольников
        # чтобы была площадь под прямоугольником надо добавить precisions[1:-1] = precisions[2:]
        average_precisions.append(np.trapz(precisions, recalls))
        average_precisions_classes[c] = np.trapz(precisions, recalls)

    return round(sum(average_precisions) / len(average_precisions), 4), \
           round(np.mean(precisions), 4),\
           round(np.mean(recalls), 4), average_precisions_classes


if __name__ == "__main__":
    pred_boxes = [[0, 0.9, 0, 0, 100, 100], [0, 0.8, 200, 200, 220, 220]]
    true_boxes = [[0, 0, 0, 100, 100], [0, 200, 200, 220, 220]]
    print(mean_average_precision(pred_boxes, true_boxes))
