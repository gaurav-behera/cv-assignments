from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align

from . import _utils as det_utils

def compute_theta_loss(preds, targets):
    # print(f"{preds.shape=} {targets.shape=}")
    # print(f"{preds.device=}, {targets.device=}")
    num_bins = preds.shape[1]
    if num_bins == 1:
        # regression
        return F.mse_loss(preds[:,0], targets)
    else:
        # classification
        bin_size = torch.pi / num_bins
        targets_bins = (targets / bin_size).long()
        targets_bins = torch.clamp(targets_bins, 0, num_bins - 1)
        return F.cross_entropy(preds, targets_bins)

def fastrcnn_loss(class_logits, box_regression, theta_preds, labels, regression_targets, theta_targets):
    # type: (Tensor, Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
        theta_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
        theta_loss (Tensor)
    """
    # print(f"{class_logits.shape=} {box_regression.shape=} {theta_preds.shape=}")
    # print(f"{labels[0].shape=} {regression_targets[0].shape=} {theta_targets[0].shape=}")
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    theta_targets = torch.cat(theta_targets, dim=0)
    # print(f"{labels.shape=} {regression_targets.shape=} {theta_targets.shape=}")

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()
    
    # theta loss
    preds = theta_preds[sampled_pos_inds_subset.to(theta_preds.device)][:,1:]
    targets = theta_targets[sampled_pos_inds_subset.to(theta_targets.device)].to(preds.device)
    
    theta_loss = compute_theta_loss(preds, targets)
    
    return classification_loss, box_loss, theta_loss

def fastrcnn_theta_loss(theta_preds, theta_targets):
    print(f"{len(theta_preds)=}")
    print(f"{len(theta_targets)=}")
    print(f"{theta_preds[0].shape=}")
    print(f"{theta_targets[0].shape=}")
    return 0

def convert_xyxytheta_to_xywha(boxes):
    xc = (boxes[:, 0] + boxes[:, 2]) / 2
    yc = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    theta = torch.deg2rad(boxes[:, 4])
    return torch.stack([xc, yc, w, h, theta], dim=1)

class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # print(f"{gt_boxes_in_image.shape=}")
                # print(f"{proposals_in_image.shape=}")
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                # match_quality_matrix = box_iou_rotated(convert_xyxytheta_to_xywha(gt_boxes_in_image), convert_xyxytheta_to_xywha(proposals_in_image))
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        # print(f"{len(proposals)=}")
        # print(f"{len(gt_boxes)=}")
        # print(f"{proposals[0].shape=}")
        # print(f"{gt_boxes[0].shape=}")
        # print(f"{proposals[0]=}")
        # proposals_with_theta = []
        # for proposal in proposals:
        #     proposal_with_theta = torch.cat((proposal, torch.zeros((proposal.shape[0], 1), dtype=proposal.dtype, device=proposal.device)), dim=1)
        #     proposals_with_theta.append(proposal_with_theta)
        # gt_boxes_without_theta = [gt_box[:, :-1] for gt_box in gt_boxes]
        # print(f"{len(proposal_with_theta)=}")
        # print(f"{proposals_with_theta[0].shape=}")
        # print(f"{proposals_with_theta[0]=}")
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_thetas = [t["thetas"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        matched_gt_thetas = []
        
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            gt_thetas_in_image = gt_thetas[img_id]
            
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            matched_gt_thetas.append(gt_thetas_in_image[matched_idxs[img_id].to(gt_thetas_in_image.device)])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        theta_targets = matched_gt_thetas
        return proposals, matched_idxs, labels, regression_targets, theta_targets

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        theta_preds,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_theta_list = theta_preds.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_thetas = []
        for boxes, scores, thetas, image_shape in zip(pred_boxes_list, pred_scores_list, pred_theta_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            thetas = thetas[:, 1:]
            
            if thetas.shape[1] != 1:
                nbins = thetas.shape[1]
                angle_per_bin = torch.pi / nbins
                max_val_idx = torch.argmax(thetas, dim=1)
                max_val_theta = angle_per_bin * max_val_idx.float()
            
            thetas = max_val_theta

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            thetas = thetas.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, thetas = boxes[inds], scores[inds], labels[inds], thetas[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, thetas = boxes[keep], scores[keep], labels[keep], thetas[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, thetas = boxes[keep], scores[keep], labels[keep], thetas[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_thetas.append(thetas)

        return all_boxes, all_scores, all_labels, all_thetas

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        # targets_without_theta = []
        # if targets is not None:
        #     for target in targets:
        #         target_without_theta = {"boxes": target["boxes"][:, :-1], "labels": target["labels"]}
        #         targets_without_theta.append(target_without_theta)
        if self.training:
            proposals, matched_idxs, labels, regression_targets, theta_targets = self.select_training_samples(proposals, targets)
            # print("---------")
            # print(f"{theta_targets.shape=}")
        else:
            labels = None
            regression_targets = None
            theta_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        
        class_logits, box_regression, theta_preds = self.box_predictor(box_features)
        # print(f"{class_logits.shape=}")
        # print(f"{box_regression.shape=}")
        # print(f"{theta_preds.shape=}")

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg, loss_theta = fastrcnn_loss(class_logits, box_regression, theta_preds, labels, regression_targets, theta_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_theta": loss_theta}
        else:
            boxes, scores, labels, thetas = self.postprocess_detections(class_logits, box_regression, theta_preds, proposals, image_shapes)
            # print(f"{scores[0]=}")
            # print(f"{labels[0]=}")
            # print(f"{thetas[0]=}")
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "thetas": thetas[i]
                    }
                )
                # print(f"{result}")

        return result, losses
    