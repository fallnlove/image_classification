from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.metrics.tracker import MetricTracker


class Inferencer:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        dataloaders: dict,
        transforms: dict = None,
    ):
        self.model = model

        self.device = device
        self.dataloaders = dataloaders

        self.transforms = transforms

        self.save_path = Path("./submissions/")
        if not self.save_path.exists():
            self.save_path.mkdir()

    def run(self):
        """
        Run training process.
        """

        self.table = []
        try:
            self._eval_process()
        except KeyboardInterrupt as e:
            print("Keyboard interrupt. Saving checkpoint.")
            self._save_checkpoint(-1)

        self.table.sort(key=lambda x, y: x["Id"] < y["Id"])
        preds = pd.DataFrame(self.table, columns=["Id", "Category"])
        preds.to_csv(str(self.save_path / "sample_submission.csv"))

    @torch.no_grad()
    def _eval_process(self):
        """
        Start inference.
        """
        self.model.eval()

        for index, batch in tqdm(
            enumerate(self.dataloaders["test"]), total=len(self.dataloaders["test"])
        ):
            batch = self._process_batch(batch, self.train_tracker)
            self._log_batch(batch)

    def _process_batch(self, batch, tracker):
        batch = self._move_to_device(batch)
        batch = self._transform_batch(batch)

        output = self.model(**batch)
        batch.update(output)

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

        transform = self.transforms["test"]

        batch["images"] = transform(batch["images"])

        return batch

    def _log_batch(self, batch: dict):
        """
        Log batch of data.
        Input:
            batch (dict): batch of data.
        """

        logits = batch["predictions"].argmax(-1).detach().cpu().numpy()
        for name, logit in zip(batch["file_names"], logits):
            self.table.append(
                {
                    "Id": name,
                    "Category": logit,
                }
            )
