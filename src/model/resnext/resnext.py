from torch import Tensor, nn
from torchvision.models import resnext50_32x4d


class ResNext(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNext, self).__init__()

        self.model = resnext50_32x4d(*args, **kwargs)

    def forward(self, images: Tensor, **batch) -> dict[Tensor]:
        """
         Input:
            images (Tensor): input image (B, input_dim, H, W).
        Output:
            predictions (Tensor): predictions (B, num_classes)
        """

        predictions = self.model(images)

        return {"predictions": predictions}
