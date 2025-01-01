from torch import Tensor, nn
from torchvision.models import wide_resnet50_2


class WideResNet(nn.Module):
    def __init__(self, type, *args, **kwargs):
        assert type in ["small", "medium", "large"]
        super(WideResNet, self).__init__()

        self.model = wide_resnet50_2(*args, **kwargs)

    def forward(self, images: Tensor, **batch) -> dict[Tensor]:
        """
         Input:
            images (Tensor): input image (B, input_dim, H, W).
        Output:
            predictions (Tensor): predictions (B, num_classes)
        """

        predictions = self.model(images)

        return {"predictions": predictions}
