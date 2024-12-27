from collections import defaultdict
from typing import Union

import torch


def collate_fn(dataset_items: list[dict]) -> dict[Union[torch.Tensor, list]]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Union[Tensor, list]]): dict, containing batch-version
            of the tensors.
    """

    result_batch = defaultdict()

    result_batch["images"] = torch.stack([item["image"] for item in dataset_items])

    result_batch["file_names"] = [item["file_name"] for item in dataset_items]

    if "label" in dataset_items[0].keys():
        result_batch["labels"] = torch.LongTensor(
            [item["label"] for item in dataset_items]
        )

    return result_batch
