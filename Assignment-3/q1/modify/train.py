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
import wandb

from infer import evaluate_metrics

import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    st = SceneTextDataset("train", root_dir=dataset_config["root_dir"], augment=args.flip_augment)

    train_dataset = DataLoader(
        st, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_function
    )

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        min_size=600,
        max_size=1000,
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config["num_classes"],
        num_theta_bins=args.num_theta_bins,
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
    map, precision, recall = 0,0,0

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        frcnn_theta_losses = []
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
            loss += args.theta_loss_weight_factor * batch_losses["loss_theta"]
            loss += batch_losses["loss_rpn_box_reg"]
            loss += batch_losses["loss_objectness"]

            rpn_classification_losses.append(batch_losses["loss_objectness"].item())
            rpn_localization_losses.append(batch_losses["loss_rpn_box_reg"].item())
            frcnn_classification_losses.append(batch_losses["loss_classifier"].item())
            frcnn_localization_losses.append(batch_losses["loss_box_reg"].item())
            frcnn_theta_losses.append(batch_losses["loss_theta"].item())

            loss.backward()
            optimizer.step()
            step_count += 1
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
        loss_output += " | FRCNN Theta Loss : {:.4f}".format(
            np.mean(frcnn_theta_losses)
        )
        # print(loss_output)
        
        args2 = argparse.Namespace()
        args2.config_path = args.config_path
        args2.split_type = "val"
        args2.num_theta_bins = args.num_theta_bins
        map, precision, recall = evaluate_metrics(args2)
        print("mAP: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(map, precision, recall))
        if args.log_wandb:
            wandb.log({
                "mAP": map,
                "Precision": precision,
                "Recall": recall,
                "RPN Classification Loss": np.mean(rpn_classification_losses),
                "RPN Localization Loss": np.mean(rpn_localization_losses),
                "FRCNN Classification Loss": np.mean(frcnn_classification_losses),
                "FRCNN Localization Loss": np.mean(frcnn_localization_losses),
                "FRCNN Theta Loss": np.mean(frcnn_theta_losses),
            })
    print("Done Training...")
    return map, precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for faster rcnn using torchvision code training"
    )
    parser.add_argument(
        "--config", dest="config_path", default="config/st.yaml", type=str
    )
    args = parser.parse_args()
    args.flip_augment = False

    # sweep_config = {
    #     "method": "grid",  
    #     "parameters": {
    #         "weight_factor": {
    #             "values": [0.5, 1.0, 10.0]
    #         },
    #         "num_theta_bins": {
    #             "values": [3, 10, 1]
    #         },
    #     },
    # }
    # run_id = 0
    # def sweep_train():
    #     global run_id
    #     args.log_wandb = True
        
    #     wandb.init(project="oriented_fasterrcnn", name=f"run_{run_id}")
    #     run_id += 1
    #     config = wandb.config

    #     args.theta_loss_weight_factor = config.weight_factor
    #     args.num_theta_bins = config.num_theta_bins
    #     map, precision, recall = train(args)
        
    #     if args.log_wandb:
    #         wandb.log({
    #             "mAP": map,
    #             "Precision": precision,
    #             "Recall": recall,
    #         })
    #     wandb.finish()

    # sweep_id = wandb.sweep(sweep_config, project="oriented_fasterrcnn")
    # wandb.agent(sweep_id, function=sweep_train, count=9)  

    args.log_wandb = False
    args.theta_loss_weight_factor = 1.0
    args.num_theta_bins = 10
    # args.flip_augment = True
    train(args)
    

