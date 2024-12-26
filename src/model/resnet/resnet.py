import torch
from torch import Tensor, nn

from src.model.resnet.basic_block import BasicBlock


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        num_blocks: int,
        num_classes: int = 200,
        hidden_dim: int = 16,
    ):
        """
        Input:
            block (nn.Module): type of block to use in ResNet.
            num_blocks (int): number of blocks.
            num_classes (int): number of classes in classification.
            hidden_dim (int): hidden dim.
        """
        super(self, ResNet).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            self._make_layer(block, hidden_dim, num_blocks[0], 1),
            self._make_layer(block, hidden_dim * 2, num_blocks[1], 2),
            self._make_layer(block, hidden_dim * 4, num_blocks[2], 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def forward(self, images: Tensor, **batch) -> dict[Tensor]:
        """
         Input:
            images (Tensor): input image (B, input_dim, H, W).
        Output:
            predictions (Tensor): predictions (B, num_classes)
        """

        predictions = self.body(images)

        return {"predictions": predictions}


class ResNet20(ResNet):
    def __init__(self, *args, **kwargs):
        super(self, ResNet20).__init__(BasicBlock, [3, 3, 3], *args, **kwargs)


class ResNet110(nn.Module):
    def __init__(self, *args, **kwargs):
        super(self, ResNet110).__init__(BasicBlock, [18, 18, 18], *args, **kwargs)
