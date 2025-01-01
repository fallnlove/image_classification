from torch import Tensor, nn
from torchvision.models import wide_resnet50_2
from torchvision.models.resnet import Bottleneck, _resnet


class WideResNet(nn.Module):
    def __init__(self, **kwargs):
        super(WideResNet, self).__init__()

        self.model = _resnet(Bottleneck, [2, 2, 2, 2], None, True, **kwargs)

    def forward(self, images: Tensor, **batch) -> dict[Tensor]:
        """
         Input:
            images (Tensor): input image (B, input_dim, H, W).
        Output:
            predictions (Tensor): predictions (B, num_classes)
        """

        predictions = self.model(images)

        return {"predictions": predictions}
