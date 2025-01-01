from torch import Tensor, nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()

        self.model = resnet18(*args, **kwargs)

    def forward(self, images: Tensor, **batch) -> dict[Tensor]:
        """
         Input:
            images (Tensor): input image (B, input_dim, H, W).
        Output:
            predictions (Tensor): predictions (B, num_classes)
        """

        predictions = self.model(images)

        return {"predictions": predictions}
