from torch import Tensor, nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    def __init__(self, type: str = "resnet18", kernel_size: int = 3, *args, **kwargs):
        super(ResNet, self).__init__()
        assert type in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

        if type == "resnet18":
            self.model = resnet18(*args, **kwargs)
        elif type == "resnet34":
            self.model = resnet34(*args, **kwargs)
        elif type == "resnet50":
            self.model = resnet50(*args, **kwargs)
        elif type == "resnet101":
            self.model = resnet101(*args, **kwargs)
        elif type == "resnet152":
            self.model = resnet152(*args, **kwargs)

        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=kernel_size, padding="same", bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 4, 200),
        )

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
