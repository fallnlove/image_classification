from torch import Tensor, nn
from torchvision.models.resnet import Bottleneck, _resnet


class ResNext(nn.Module):
    def __init__(
        self, groups: int = 32, width_per_group: int = 8, kernel_size: int = 3, **kwargs
    ):
        super(ResNext, self).__init__()

        self.model = _resnet(
            Bottleneck,
            [2, 2, 2, 2],
            None,
            True,
            groups=groups,
            width_per_group=width_per_group,
            **kwargs
        )

        self.model.conv1 = nn.Conv2d(
            3, self.model.inplanes, kernel_size=kernel_size, padding="same", bias=False
        )
        self.model.maxpool = nn.Identity()

        nn.init.kaiming_normal_(
            self.model.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, images: Tensor, **batch) -> dict[Tensor]:
        """
         Input:
            images (Tensor): input image (B, input_dim, H, W).
        Output:
            predictions (Tensor): predictions (B, num_classes)
        """

        predictions = self.model(images)

        return {"predictions": predictions}
