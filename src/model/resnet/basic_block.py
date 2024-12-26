import torch
from torch import Tensor, nn


class BasicBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        """
        Input:
            input_dim (int): number of input channels.
            output_dim (int): number of output channels.
            stride (int): stride in convolution.
        """

        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_dim),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(output_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:
            x (Tensor): input image (B, input_dim, H, W).
        Output:
            out (Tensor): output image (B, output_dim, H, W)
        """

        out = self.layer1(x) + self.layer2(x)

        return self.relu(out)
