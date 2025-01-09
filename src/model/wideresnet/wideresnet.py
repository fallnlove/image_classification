from torch import Tensor, nn
from torchvision.models import wide_resnet50_2


class WideResNet(nn.Module):
    def __init__(self, kernel_size: int = 3, **kwargs):
        super(WideResNet, self).__init__()

        self.model = wide_resnet50_2(**kwargs)

        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=kernel_size, padding="same", bias=False
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
