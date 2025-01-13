from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.metrics.tracker import MetricTracker


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: nn.Module,
        scheduler: nn.Module,
        metrics: list,
        device: str,
        writer,
        dataloaders: dict,
        num_epochs: int,
        transforms: dict = None,
        load_checkpoint: str = None,
    ):
        self.is_train = True

        self.model = model

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

        self.device = device
        self.writer = writer
        self.dataloaders = dataloaders
        self.num_epochs = num_epochs
        self.start_epoch = 0

        self.transforms = transforms

        self.train_tracker = MetricTracker(
            [metric.name for metric in self.metrics] + ["grad_norm", criterion.name]
        )

        self.eval_tracker = MetricTracker(
            [metric.name for metric in self.metrics] + [criterion.name]
        )

        self.save_path = Path("./models/")
        if not self.save_path.exists():
            self.save_path.mkdir()

        if load_checkpoint is not None:
            self._resume_checkpoint(load_checkpoint)

    def run(self):
        """
        Run training process.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            print("Keyboard interrupt. Saving checkpoint.")
            self._save_checkpoint(-1)

    def _train_process(self):
        """
        Start training process.
        """

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self._train_epoch(epoch)
            self._eval_epoch(epoch)

            if epoch % 5 == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch):
        """
        Train epoch.
        """
        self.is_train = True
        self.train_tracker.reset()
        self.writer.log_scalar("epoch", epoch)
        self.writer.train()
        self.model.train()

        for index, batch in tqdm(
            enumerate(self.dataloaders["train"]), total=len(self.dataloaders["train"])
        ):
            batch = self._process_batch(batch, self.train_tracker)

            self.train_tracker.update("grad_norm", self._get_grad_norm())

            if index % 100 == 0:
                self.writer.log_scalar("learning rate", self.scheduler.get_last_lr()[0])
                self.writer.log_metrics(self.train_tracker)
                self._log_batch(batch)
                self.train_tracker.reset()

    @torch.no_grad()
    def _eval_epoch(self, epoch):
        self.is_train = False
        self.eval_tracker.reset()
        self.writer.eval()
        self.model.eval()

        for batch in tqdm(
            self.dataloaders["eval"], total=len(self.dataloaders["eval"])
        ):
            batch = self._process_batch(batch, self.eval_tracker)

        self.writer.log_metrics(self.eval_tracker)
        self._log_batch(batch)

    def _process_batch(self, batch, tracker):
        batch = self._move_to_device(batch)
        batch = self._transform_batch(batch)

        if self.is_train:
            self.optimizer.zero_grad()

        output = self.model(**batch)
        batch.update(output)

        loss = self.criterion(**batch)
        batch.update(loss)

        if self.is_train:
            batch["loss"].backward()
            self.optimizer.step()
            self.writer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        tracker.update(self.criterion.name, batch["loss"].item())

        for metric in self.metrics:
            tracker.update(metric.name, metric(**batch))

        return batch

    def _move_to_device(self, batch: dict):
        """
        Move batch to device. Batch should have a key "images" which is the input data.

        Input:
            batch (dict): batch of data
        Output:
            batch (dict): batch of data
        """
        batch["images"] = batch["images"].to(self.device)
        batch["labels"] = batch["labels"].to(self.device)

        return batch

    def _transform_batch(self, batch: dict):
        """
        Transform batch of data.

        Input:
            batch (dict): batch of data
        Output:
            batch (dict): batch of data
        """
        if self.transforms is None:
            return batch

        transform = (
            self.transforms["train"] if self.is_train else self.transforms["eval"]
        )

        batch["images"] = transform(batch["images"])

        return batch

    @torch.no_grad()
    def _get_grad_norm(self):
        """
        Get gradient norm of the model.
        """

        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        return total_norm

    def _log_batch(self, batch: dict):
        """
        Log batch of data.
        Input:
            batch (dict): batch of data.
        """

        image = batch["images"][0].detach().cpu().numpy().transpose(1, 2, 0)
        self.writer.log_image("image", image)

    def _save_checkpoint(self, epoch: int):
        """
        Save model checkpoint.
        Input:
            epoch (int): epoch number.
        """
        save_path = self.save_path / f"model_{epoch}.pth"

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        torch.save(state, save_path)

    def _resume_checkpoint(self, path: str):
        """
        Save model checkpoint.
        Input:
            epoch (int): epoch number.
        """

        checkpoint = torch.load(path, self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
