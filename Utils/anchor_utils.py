import numpy as np
from .data_prepair import compute_bbox_area, compute_box_center, compute_box_width_height


def generate_anchor_boxes(img_size, anchor_sizes, anchor_ratios, stride=1):  # TODO: stride must be (stride_x, stride_y)
    input_height, input_width = img_size
    num_sizes, num_ratios = len(anchor_sizes), len(anchor_ratios)
    num_boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = np.array(anchor_sizes)
    ratio_tensor = np.array(anchor_ratios)

    anchor_centers = generate_anchor_centers(input_height, input_width, num_boxes_per_pixel, stride)

    anchor_wh = generate_anchor_wh(input_height, input_width, ratio_tensor, size_tensor, stride)

    output = anchor_centers + anchor_wh
    return np.expand_dims(output, axis=0)


def generate_anchor_wh(input_height, input_width, ratio_tensor, size_tensor, stride=1):
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]), size_tensor[0] * np.sqrt(ratio_tensor[1:]))) * \
        input_height / input_width
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]), size_tensor[0] / np.sqrt(ratio_tensor[1:])))
    # num_rep = int(np.round(input_height/stride)) * int(np.round(input_width/stride))
    num_rep = len(np.arange(0, input_height, stride)) * len(np.arange(0, input_width, stride))
    anchor_wh = np.tile(np.stack((-w, -h, w, h)).T, (num_rep, 1)) / 2
    return anchor_wh


def generate_anchor_centers(height, width, num_boxes_per_pixel, stride=1):
    offset_h = 0.5
    offset_w = 0.5
    steps_h = 1 / height
    steps_w = 1 / width
    center_h = (np.arange(0, height, stride) + offset_h) * steps_h
    center_w = (np.arange(0, width, stride) + offset_w) * steps_w
    box_center_y, box_center_x = np.meshgrid(center_h, center_w)
    box_center_y, box_center_x = box_center_y.reshape(-1), box_center_x.reshape(-1)
    anchor_centers = np.repeat(np.stack([box_center_x, box_center_y, box_center_x, box_center_y], axis=1),
                               num_boxes_per_pixel, axis=0)
    return anchor_centers


def compute_iou(anchor_boxes, ground_truth_boxes):

    anchor_areas = compute_bbox_area(anchor_boxes)
    gt_areas = compute_bbox_area(ground_truth_boxes)

    inter_upper_lefts = np.maximum(anchor_boxes[:, None, :2], ground_truth_boxes[:, :2])
    inter_lower_rights = np.minimum(anchor_boxes[:, None, 2:], ground_truth_boxes[:, 2:])
    inters = (inter_lower_rights - inter_upper_lefts).clip(min=0)

    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = anchor_areas[:, None] + gt_areas - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(anchors, ground_truth, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]

    jaccard = compute_iou(anchors, ground_truth)

    anchors_bbox_map = np.ones(num_anchors, dtype=np.int32) * -1

    max_iou, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_iou >= iou_threshold)[0]
    box_j = indices[max_iou >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def box_to_center_wh(boxes):
    cx, cy = compute_box_center(boxes)
    w, h = compute_box_width_height(boxes)
    boxes = np.stack((cx, cy, w, h), axis=-1)
    return boxes


def center_wh_to_box(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    cwh_anc = box_to_center_wh(anchors)
    cwh_assigned_bb = box_to_center_wh(assigned_bb)
    offset_xy = 10 * (cwh_assigned_bb[:, :2] - cwh_anc[:, :2]) / cwh_anc[:, 2:]
    offset_wh = 5 * np.log(eps + cwh_assigned_bb[:, 2:] / cwh_anc[:, 2:])
    offset = np.concatenate([offset_xy, offset_wh], axis=1)
    return offset


def compute_anchor_targets(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    num_anchors = anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(anchors, label[:, 1:])
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0), axis=-1)), (1, 4)).astype('int32')

        class_labels = np.zeros(num_anchors, dtype=np.int32)
        assigned_bb = np.zeros((num_anchors, 4), dtype=np.float32)

        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = np.stack(batch_offset)
    bbox_mask = np.stack(batch_mask)
    class_labels = np.stack(batch_class_labels)
    return bbox_offset, bbox_mask, class_labels


def offset_inverse(anchors, offset_predictions):
    anc = box_to_center_wh(anchors)
    pred_bbox_xy = (offset_predictions[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = np.exp(offset_predictions[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = np.concatenate((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = center_wh_to_box(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    scores_arg_sort = scores.argsort()[::-1]
    keep = []
    while scores_arg_sort.size > 0:
        i = scores_arg_sort[0]
        keep.append(i)
        if scores_arg_sort.size == 1:
            break
        iou = compute_iou(boxes[i, :].reshape(-1, 4), boxes[scores_arg_sort[1:], :].reshape(-1, 4)).reshape(-1)
        indices = np.nonzero(iou <= iou_threshold)[0]
        scores_arg_sort = scores_arg_sort[indices + 1]
    return np.array(keep, dtype=np.int32)


def box_detection(cls_probs, offset_predictions, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    batch_size = cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_predictions[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        all_idx = np.arange(num_anchors, dtype=np.int32)
        combined = np.concatenate((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = np.concatenate((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]

        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = np.concatenate((np.expand_dims(class_id, axis=1), np.expand_dims(conf, axis=1), predicted_bb),
                                   axis=1)
        out.append(pred_info)
    return np.stack(out)
