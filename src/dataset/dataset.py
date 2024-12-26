import random
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(
        self,
        path: str,
        part: str,
        shuffle_index: bool = False,
        instance_transforms=None,
    ):
        """
        Input:
            path (str): path to dataset.
            part (str): part of dataset (train, val or test).
            shuffle_index (bool): shuffle dataset.
            instance_transforms (Callable): augmentations.
        """
        assert part in ["train", "val", "test"]

        self.dataset_path = Path(path).absolute().resolve()
        self.part = part
        self.shuffle_index = shuffle_index
        self.instance_transforms = instance_transforms

        self.base = self._get_base()

        if shuffle_index:
            random.seed(42)
            random.shuffle(self.base)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        info = self.base[index]

        data = {"image": self.load_image(info["path"])}
        if self.part != "test":
            data["label"] = info["label"]

        if self.instance_transforms is not None:
            data["image"] = self.instance_transforms(data["image"])

        return data

    def load_image(self, path):
        image = read_image(str(path), mode=ImageReadMode.RGB).float() / 255.0

        return image

    def _get_base(self):
        path = self.dataset_path / self.part
        labels = pd.read_csv(self.dataset_path / "labels.csv", index_col=0)

        data = []

        for file_path in tqdm(path.iterdir()):
            info = {
                "path": file_path,
            }
            if self.part != "test":
                info["label"] = labels.loc[file_path.name].item()

            data.append(info)

        return data
