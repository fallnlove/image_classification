from torch import Tensor, nn
from torchvision.models.resnet import Bottleneck, _resnet


class ResNext(nn.Module):
    def __init__(self, **kwargs):
        super(ResNext, self).__init__()

        self.model = _resnet(
            Bottleneck, [2, 2, 2, 2], None, True, groups=32, width_per_group=8, **kwargs
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
