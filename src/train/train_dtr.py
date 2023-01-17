import torch

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import sys, os
from os.path import dirname
from datetime import datetime
import copy
import argparse

module_path = os.path.abspath(dirname(dirname(dirname(__file__))))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.models import build_model
from src.config.configs import ConfigLoader
from src.datasets.igibson_dataset import IGibsonDataset
from src.utils.data_utils import igibson_collate


def train_dtr(cfg, device, data_path, log_path, model_path="none"):
    model, criterion, _ = build_model(cfg)

    if model_path != "none":
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = IGibsonDataset(data_path, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        # num_workers=cfg["num_workers"],
        collate_fn=igibson_collate,
    )

    print("start training")
    print("num_queries: {}".format(cfg.num_queries))
    nb_epochs = cfg.train.epochs

    torch.save(model.state_dict(), f"{log_path}/model_start.pth")

    for epoch in range(nb_epochs + 1):
        step = 0
        for sample_temp in dataloader:
            sample = copy.deepcopy(sample_temp)
            del sample_temp
            step += 1
            img, gt = sample
            img = img.to(device)
            for target in gt:
                target["objects"] = target["objects"].to(device)

            pred = model(img)

            # check nan
            contain_nan = False
            for key in pred:
                if key == "aux_outputs":
                    break
                if not (~torch.isnan(pred[key])).all():
                    contain_nan = True
                    break
            if contain_nan:
                print("nan produced")
                continue

            loss_dict = criterion(pred, gt, dataset.angle_included_categories)
            weight_dict = criterion.weight_dict
            loss = (
                weight_dict["loss_ce"] * loss_dict["loss_ce"]
                + weight_dict["loss_bbox"] * loss_dict["loss_bbox"]
                + weight_dict["loss_giou"] * loss_dict["loss_giou"]
                + weight_dict["loss_angle"] * loss_dict["loss_angle"]
                + weight_dict["loss_offset"] * loss_dict["loss_offset"]
                + weight_dict["loss_size"] * loss_dict["loss_size"]
                + weight_dict["loss_depth"] * loss_dict["loss_depth"]
            )
            loss_dict["train_total"] = loss

            scaler.scale(loss).backward()

            if step % 10 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                print("-" * 100)
                print(
                    "iter:",
                    "%04d" % (step),
                    "total: {:.9f}, ce: {:.9f}, bbox: {:.9f}, giou: {:.9f}, angle: {:.9f}, offset: {:.9f}, size: {:.9f}, depth: {:.9f}".format(
                        loss,
                        loss_dict["loss_ce"],
                        loss_dict["loss_bbox"],
                        loss_dict["loss_giou"],
                        loss_dict["loss_angle"],
                        loss_dict["loss_offset"],
                        loss_dict["loss_size"],
                        loss_dict["loss_depth"],
                    ),
                )
                print("-" * 100)

        print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(loss))
        torch.save(model.state_dict(), f"{log_path}/model_{epoch}.pth")


def train_one_img(cfg, device, data_path, log_path):
    model, criterion, _ = build_model(cfg)
    scaler = GradScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    print("start training")
    nb_epochs = 1000

    torch.save(model.state_dict(), f"{log_path}/model_start.pth")

    step = 0
    item_id = 1000
    dataset = IGibsonDataset(data_path, cfg)

    for epoch in range(nb_epochs + 1):
        step += 1
        optimizer.zero_grad()
        img, gt = dataset.__getitem__(item_id)
        img = img.reshape((1, 3, 720, 1280)).to(device)

        gt["objects"] = gt["objects"].to(device)
        gt = [gt]
        pred = model(img)
        loss_dict = criterion(pred, gt, dataset.angle_included_categories)
        weight_dict = criterion.weight_dict
        loss = (
            weight_dict["loss_ce"] * loss_dict["loss_ce"]
            + weight_dict["loss_bbox"] * loss_dict["loss_bbox"]
            # + weight_dict["loss_angle"] * loss_dict["loss_angle"]
            # + weight_dict["loss_offset"] * loss_dict["loss_offset"]
            # + weight_dict["loss_size"] * loss_dict["loss_size"]
            # + weight_dict["loss_depth"] * loss_dict["loss_depth"]
        )
        loss_dict["train_total"] = loss

        scaler.scale(loss).backward()

        if step % 20 == 0:
            # train_loss = {f"train_{k}": v for k, v in loss_dict.items()}
            scaler.step(optimizer)
            scaler.update()
            print("-" * 100)
            print(
                "iter:",
                "%04d" % (step),
                "total: {:.9f}, ce: {:.9f}, bbox: {:.9f}, angle: {:.9f}, offset: {:.9f}, size: {:.9f}, depth: {:.9f}".format(
                    loss,
                    loss_dict["loss_ce"],
                    loss_dict["loss_bbox"],
                    loss_dict["loss_angle"],
                    loss_dict["loss_offset"],
                    loss_dict["loss_size"],
                    loss_dict["loss_depth"],
                ),
            )
            print("-" * 100)
        torch.save(model.state_dict(), f"{log_path}/model_{epoch}.pth")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--cfg_path", default="ODAM/configs/detr_scan_net.yaml")
    args = arg_parser.parse_args()

    cfg_path = args.cfg_path
    data_path = "ODAM/src/datasets/"
    cfg = ConfigLoader().merge_cfg([cfg_path])
    log_path = os.path.join(
        "ODAM/src/train/log/",
        datetime.now().strftime("%y%m%d%H%M%S%f")[:-4],
    )
    os.mkdir(log_path)
    print("saving at {}".format(log_path))

    model_path = "ODAM/src/train/log/23011216251146/model_40.pth"

    device = cfg.device
    # train_one_img(cfg, device, data_path, log_path)
    train_dtr(cfg, device, data_path, log_path, model_path)


if __name__ == "__main__":
    main()
