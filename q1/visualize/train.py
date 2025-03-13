import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import detection
from detection.faster_rcnn import FastRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.8, device=0)

def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config["dataset_params"]
    train_config = config["train_params"]

    seed = train_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    st = SceneTextDataset("train", root_dir=dataset_config["root_dir"])
    st_val = SceneTextDataset("val", root_dir=dataset_config["root_dir"])

    train_dataset = DataLoader(
        st, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_function
    )
    val_dataset = DataLoader(
        st_val, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_function
    )

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        min_size=600,
        max_size=1000,
        rpn_post_nms_top_n_train=args.rpn_post_nms_top_n_train,
        rpn_fg_iou_thresh=args.rpn_fg_iou_thresh,
        rpn_positive_fraction=args.rpn_positive_fraction,
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config["num_classes"],
    )

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(train_config["task_name"]):
        os.mkdir(train_config["task_name"])

    optimizer = torch.optim.SGD(
        lr=1e-4,
        params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        weight_decay=5e-5,
        momentum=0.9,
    )

    num_epochs = train_config["num_epochs"]
    step_count = 0

    # create directories for saving outputs
    os.makedirs(f"outputs/{args.hyperparam_idx}/objectness/")
    os.makedirs(f"outputs/{args.hyperparam_idx}/object_proposals/")
    os.makedirs(f"outputs/{args.hyperparam_idx}/bb_assignments/")
    os.makedirs(f"outputs/{args.hyperparam_idx}/roi_head_outputs/")

    # set hyperparameter index for saving
    faster_rcnn_model.rpn.hyperparam_idx = args.hyperparam_idx
    faster_rcnn_model.roi_heads.hyperparam_idx = args.hyperparam_idx

    # save validation images
    if args.hyperparam_idx == 1:
        os.makedirs("outputs/images/", exist_ok=True)
        for i, (ims, targets, _) in enumerate(val_dataset):
            torchvision.utils.save_image(ims[0], f"outputs/images/image_{i}.jpg")

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []

        faster_rcnn_model.rpn.val = False
        faster_rcnn_model.roi_heads.val = False
        faster_rcnn_model.rpn.epoch_idx = i
        faster_rcnn_model.roi_heads.epoch_idx = i

        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target["boxes"] = target["bboxes"].float().to(device)
                del target["bboxes"]
                target["labels"] = target["labels"].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses["loss_classifier"]
            loss += batch_losses["loss_box_reg"]
            loss += batch_losses["loss_rpn_box_reg"]
            loss += batch_losses["loss_objectness"]

            rpn_classification_losses.append(batch_losses["loss_objectness"].item())
            rpn_localization_losses.append(batch_losses["loss_rpn_box_reg"].item())
            frcnn_classification_losses.append(batch_losses["loss_classifier"].item())
            frcnn_localization_losses.append(batch_losses["loss_box_reg"].item())

            loss.backward()
            optimizer.step()
            step_count += 1

        faster_rcnn_model.rpn.val = True
        faster_rcnn_model.roi_heads.val = True

        for ims, targets, _ in tqdm(val_dataset):
            for target in targets:
                target["boxes"] = target["bboxes"].float().to(device)
                del target["bboxes"]
                target["labels"] = target["labels"].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses["loss_classifier"]
            loss += batch_losses["loss_box_reg"]
            loss += batch_losses["loss_rpn_box_reg"]
            loss += batch_losses["loss_objectness"]
        print("Validation Loss: ", loss.item())

        print("Finished epoch {}".format(i))
        torch.save(
            faster_rcnn_model.state_dict(),
            os.path.join(
                train_config["task_name"],
                "tv_frcnn_r50fpn_" + train_config["ckpt_name"],
            ),
        )
        loss_output = ""
        loss_output += "RPN Classification Loss : {:.4f}".format(
            np.mean(rpn_classification_losses)
        )
        loss_output += " | RPN Localization Loss : {:.4f}".format(
            np.mean(rpn_localization_losses)
        )
        loss_output += " | FRCNN Classification Loss : {:.4f}".format(
            np.mean(frcnn_classification_losses)
        )
        loss_output += " | FRCNN Localization Loss : {:.4f}".format(
            np.mean(frcnn_localization_losses)
        )
        print(loss_output)
    print("Done Training...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for faster rcnn using torchvision code training"
    )
    parser.add_argument(
        "--config", dest="config_path", default="config/st.yaml", type=str
    )
    args = parser.parse_args()

    # os.system("rm -r outputs/1/*")
    # os.system("rm -r outputs/2/*")
    # os.system("rm -r outputs/3/*")

    # different hyperparameters
    rpn_post_nms_top_n_train_vals = [2000, 1000, 500]
    rpn_fg_iou_thresh_vals = [0.8, 0.7, 0.5]
    rpn_positive_fraction_vals = [0.8, 0.5, 0.3]
    for i, (x, y, z) in enumerate(
        zip(
            rpn_post_nms_top_n_train_vals,
            rpn_fg_iou_thresh_vals,
            rpn_positive_fraction_vals,
        )
    ):
        args.rpn_post_nms_top_n_train = x
        args.rpn_fg_iou_thresh = y
        args.rpn_positive_fraction = z
        args.hyperparam_idx = i + 1
        train(args)
