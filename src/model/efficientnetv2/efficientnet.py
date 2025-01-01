from torch import Tensor, nn
from torchvision.models import efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s


class EfficientNetV2(nn.Module):
    def __init__(self, type, *args, **kwargs):
        assert type in ["small", "medium", "large"]
        super(EfficientNetV2, self).__init__()

        if type == "small":
            self.model = efficientnet_v2_s(*args, **kwargs)
        elif type == "medium":
            self.model = efficientnet_v2_m(*args, **kwargs)
        else:
            self.model = efficientnet_v2_l(*args, **kwargs)

    def forward(self, images: Tensor, **batch) -> dict[Tensor]:
        """
         Input:
            images (Tensor): input image (B, input_dim, H, W).
        Output:
            predictions (Tensor): predictions (B, num_classes)
        """

        predictions = self.model(images)

        return {"predictions": predictions}
