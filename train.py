import argparse

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CustomDataset
from src.dataset.collate import collate_fn
from src.loss import CrossEntropyLossWrapper
from src.metrics import Accuracy
from src.model import ResNet, ResNext
from src.optimizer import SAM
from src.scheduler import WarmupLR
from src.trainer import Trainer
from src.utils import set_random_seed
from src.writer import EmptyWriter, WanDBWriter


def main(config):
    set_random_seed(112)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(type="resnet50", num_classes=200)
    model = model.to(device)

    dataset_train = CustomDataset(config["path"], "train")
    dataset_val = CustomDataset(config["path"], "val")

    configs = {
        "num_epochs_1": config["numepochs1"],
        "num_epochs_2": config["numepochs2"],
        "warmup_epochs": config["warmupepochs"],
        "lr_1": config["lr1"],
        "lr_2": config["lr2"],
        "weight_decay": config["weightdecay"],
        "momentum": config["momentum"],
        "nesterov": config["nesterov"],
        "batch_size": config["batchsize"],
        "model": "ResNet50",
        "optimizer": "SAM(SGD)",
        "scheduler": "CosineAnnealingLR",
        "augs": "trivial+rand,trivial+rand+augmix",
    }

    train_loader = DataLoader(
        dataset_train,
        batch_size=configs["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=configs["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    writer = WanDBWriter(project_name="DL-HW-1", config=configs)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    loss_fn = CrossEntropyLossWrapper(label_smoothing=0.1).to(device)
    optimizer = SAM(
        trainable_params,
        torch.optim.SGD,
        lr=configs["lr_1"],
        weight_decay=configs["weight_decay"],
        momentum=configs["momentum"],
        nesterov=configs["nesterov"],
    )
    metrics = [Accuracy()]
    scheduler = WarmupLR(
        optimizer, warmup_steps=configs["warmup_epochs"] * len(train_loader)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandAugment(),
            transforms.TrivialAugmentWide(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.569, 0.545, 0.493), std=(0.2387, 0.2345, 0.251)
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.569, 0.545, 0.493), std=(0.2387, 0.2345, 0.251)
            ),
        ]
    )
    first_trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=device,
        writer=writer,
        dataloaders={
            "train": train_loader,
            "eval": val_loader,
        },
        num_epochs=configs["num_epochs_1"],
        transforms={
            "train": transform_train,
            "eval": transform_test,
        },
    )
    first_trainer.run()

    transform_train = transforms.Compose(
        [
            transforms.RandAugment(),
            transforms.TrivialAugmentWide(),
            transforms.AugMix(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.569, 0.545, 0.493), std=(0.2387, 0.2345, 0.251)
            ),
        ]
    )
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=configs["lr_2"],
        weight_decay=configs["weight_decay"],
        momentum=configs["momentum"],
        nesterov=configs["nesterov"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=configs["num_epochs_2"] * len(train_loader)
    )
    second_trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=device,
        writer=writer,
        dataloaders={
            "train": train_loader,
            "eval": val_loader,
        },
        num_epochs=configs["num_epochs_2"],
        transforms={
            "train": transform_train,
            "eval": transform_test,
        },
    )
    second_trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-path",
        "--path",
        default="./",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "-wandb",
        "--wandb",
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="log info to wandb",
    )
    parser.add_argument(
        "-numepochs1",
        "--numepochs1",
        default=51,
        type=int,
        help="Number of first training epochs",
    )
    parser.add_argument(
        "-numepochs2",
        "--numepochs2",
        default=51,
        type=int,
        help="Number of second training epochs",
    )
    parser.add_argument(
        "-warmupepochs",
        "--warmupepochs",
        default=5,
        type=int,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "-lr1",
        "--lr1",
        default=1e-1,
        type=float,
        help="Learning rate for first training",
    )
    parser.add_argument(
        "-lr2",
        "--lr2",
        default=1e-1,
        type=float,
        help="Learning rate for second training",
    )
    parser.add_argument(
        "-weightdecay",
        "--weightdecay",
        default=1e-4,
        type=float,
        help="Weight decay for SGD",
    )
    parser.add_argument(
        "-momentum",
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum for SGD",
    )
    parser.add_argument(
        "-nesterov",
        "--nesterov",
        default=True,
        type=bool,
        help="Enable nesterov option to SGD",
    )
    parser.add_argument(
        "-batchsize",
        "--batchsize",
        default=128,
        type=int,
        help="Batch size",
    )
    config = parser.parse_args()
    main(vars(config))
