import torch
import numpy as np
import cv2
import torchvision
import argparse
import random
import os
import yaml
from tqdm import tqdm
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import detection
from detection.faster_rcnn import FastRCNNPredictor
from shapely.geometry import Polygon
from detection.anchor_utils import AnchorGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_iou(det, gt):
    det_x, det_y, det_w, det_h, det_theta = det
    gt_x, gt_y, gt_w, gt_h, gt_theta = gt
    
    def get_rotated_box(x, y, w, h, theta):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx, dy = w / 2, h / 2
        corners = np.array([
            [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]
        ])
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])
        return Polygon(rotated_corners)
    
    det_poly = get_rotated_box(det_x, det_y, det_w, det_h, det_theta)
    gt_poly = get_rotated_box(gt_x, gt_y, gt_w, gt_h, gt_theta)
    
    if not det_poly.intersects(gt_poly):
        return 0.0
    
    intersection_area = det_poly.intersection(gt_poly).area
    union_area = det_poly.area + gt_poly.area - intersection_area + 1E-6
    return intersection_area / union_area


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method="area", return_pr=False):
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    all_aps = {}
    all_precisions = {}
    all_recalls = {}

    aps = []

    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label]
            for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets
            for im_dets_label in im_dets[label]
        ]

        # Sort by confidence score (descending)
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # Track matched GT boxes
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])

        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))

        # Process each detection
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            # Find the best-matching GT box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx

            # True Positive if IoU >= threshold & GT box is not already matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True

        # Compute cumulative sums for TP and FP
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum(tp + fp, eps)

        # Compute AP
        if method == "area":
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            for i in range(len(precisions) - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

            i = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])

        elif method == "interp":
            ap = (
                sum(
                    [
                        max(precisions[recalls >= t]) if any(recalls >= t) else 0
                        for t in np.arange(0, 1.1, 0.1)
                    ]
                )
                / 11.0
            )
        else:
            raise ValueError("Method must be 'area' or 'interp'")

        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
            all_precisions[label] = precisions.tolist()
            all_recalls[label] = recalls.tolist()
        else:
            all_aps[label] = np.nan
            all_precisions[label] = []
            all_recalls[label] = []

    mean_ap = sum(aps) / len(aps) if aps else 0.0

    if return_pr:
        return mean_ap, all_aps, all_precisions, all_recalls
    else:
        return mean_ap, all_aps


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config["dataset_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]

    seed = train_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    st = SceneTextDataset(args.split_type, root_dir=dataset_config["root_dir"])
    test_dataset = DataLoader(st, batch_size=1, shuffle=False)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        min_size=600,
        max_size=1000,
        box_score_thresh=0.7,
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config["num_classes"],
        num_theta_bins=args.num_theta_bins,
    )

    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(
        torch.load(
            os.path.join(
                train_config["task_name"],
                "tv_frcnn_r50fpn_" + train_config["ckpt_name"],
            ),
            map_location=device,
        )
    )

    return faster_rcnn_model, st, test_dataset


def evaluate_metrics(args):
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)

    gts = []
    preds = []

    for im, target, fname in tqdm(test_dataset):
        im_name = fname
        im = im.float().to(device)
        target_boxes = target["bboxes"].float().to(device)[0]
        target_labels = target["labels"].long().to(device)[0]
        target_thetas = target["thetas"].float().to(device)[0]
        frcnn_output = faster_rcnn_model(im, None)[0]

        boxes = frcnn_output["boxes"]
        labels = frcnn_output["labels"]
        scores = frcnn_output["scores"]
        thetas = frcnn_output["thetas"]

        pred_boxes = {label_name: [] for label_name in voc.label2idx}
        gt_boxes = {label_name: [] for label_name in voc.label2idx}

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            theta = thetas[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, theta, score])

        for idx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            theta = target_thetas[idx].detach().cpu().item()
            gt_boxes[label_name].append([x1, y1, x2, y2, theta])

        gts.append(gt_boxes)
        preds.append(pred_boxes)

    # Compute Mean Average Precision and Precision-Recall values
    mean_ap, all_aps, precisions, recalls = compute_map(
        preds, gts, method="interp", return_pr=True
    )

    mean_precision = 0
    mean_recall = 0
    num_classes = len(voc.idx2label)

    for idx in range(num_classes):
        class_name = voc.idx2label[idx]
        ap = all_aps[class_name]
        prec = precisions[class_name]
        rec = recalls[class_name]

        mean_precision += sum(prec) / len(prec) if len(prec) > 0 else 0
        mean_recall += sum(rec) / len(rec) if len(rec) > 0 else 0

        print(f"Class: {class_name}")
        print(
            f"  AP: {ap:.4f}, Precision: {sum(prec) / len(prec) if len(prec) > 0 else 0:.4f}, Recall: {sum(rec) / len(rec) if len(rec) > 0 else 0:.4f}"
        )

    mean_precision /= num_classes
    mean_recall /= num_classes

    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    return mean_ap, mean_precision, mean_recall


def infer(args):
    output_dir = "samples_tv_r50fpn"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)

    for sample_count in tqdm(range(10)):
        random_idx = random.randint(0, len(voc))
        im, target, fname = voc[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        # Saving images with ground truth boxes
        for idx, box in enumerate(target["bboxes"]):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            theta = target["thetas"][idx].detach().cpu().numpy() * 180 / np.pi

            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            box = cv2.boxPoints(((cx, cy), (w, h), theta))
            box = box.astype(np.int32)
            cv2.drawContours(gt_im, [box], 0, (0, 255, 0), 2)
            cv2.drawContours(gt_im_copy, [box], 0, (0, 255, 0), 2)

        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite("{}/output_frcnn_gt_{}.png".format(output_dir, sample_count), gt_im)

        # Getting predictions from trained model
        frcnn_output = faster_rcnn_model(im, None)[0]
        boxes = frcnn_output["boxes"]
        labels = frcnn_output["labels"]
        scores = frcnn_output["scores"]
        thetas = frcnn_output["thetas"]
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            theta = thetas[idx].detach().cpu().numpy() * 180 / np.pi
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            box = cv2.boxPoints(((cx, cy), (w, h), theta))
            box = box.astype(np.int32)
            cv2.drawContours(im, [box], 0, (0, 255, 0), 2)
            cv2.drawContours(im_copy, [box], 0, (0, 255, 0), 2)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite("{}/output_frcnn_{}.jpg".format(output_dir, sample_count), im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for inference using torchvision code faster rcnn"
    )
    parser.add_argument(
        "--config", dest="config_path", default="config/st.yaml", type=str
    )
    parser.add_argument("--evaluate", dest="evaluate", default=True, type=bool)
    parser.add_argument(
        "--infer_samples", dest="infer_samples", default=True, type=bool
    )
    args = parser.parse_args()
    args.split_type = "test"
    args.num_theta_bins = 10
    if args.infer_samples:
        infer(args)
    else:
        print("Not Inferring for samples as `infer_samples` argument is False")

    if args.evaluate:
        evaluate_metrics(args)
    else:
        print("Not Evaluating as `evaluate` argument is False")
