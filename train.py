import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import CustomDataset
from src.dataset.collate import collate_fn
from src.loss import CrossEntropyLossWrapper
from src.metrics import Accuracy
from src.model import EfficientNetV2
from src.scheduler import WarmupLR
from src.trainer import Trainer
from src.utils import set_random_seed
from src.writer import EmptyWriter, WanDBWriter


def main(config):
    set_random_seed(112)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetV2("small", num_classes=200)
    model = model.to(device)

    dataset_train = CustomDataset(config["path"], "train")
    dataset_val = CustomDataset(config["path"], "val")

    train_loader = DataLoader(
        dataset_train,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    num_epochs = 600

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    loss_fn = CrossEntropyLossWrapper().to(device)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.005)
    metrics = [Accuracy()]
    scheduler = torch.optim.lr_scheduler.WarmupLR(
        optimizer, warmup_steps=5 * len(train_loader)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(40),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=(0.569, 0.545, 0.493), std=(0.2387, 0.2345, 0.251)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Normalize(
                mean=(0.569, 0.545, 0.493), std=(0.2387, 0.2345, 0.251)
            ),
        ]
    )

    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=device,
        writer=WanDBWriter(project_name="DL-HW-1")
        if config["wandb"]
        else EmptyWriter(),
        dataloaders={
            "train": train_loader,
            "eval": val_loader,
        },
        num_epochs=num_epochs,
        transforms={
            "train": transform_train,
            "eval": transform_test,
        },
    )

    trainer.run()


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
    config = parser.parse_args()
    main(vars(config))
