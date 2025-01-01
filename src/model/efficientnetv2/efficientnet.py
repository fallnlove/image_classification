from torchvision.models import efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s


class EfficientNetV2:
    def __init__(self, type, *args, **kwargs):
        assert type in ["small", "medium", "large"]

        if type == "small":
            self.model = efficientnet_v2_s(*args, **kwargs)
        elif type == "medium":
            self.model = efficientnet_v2_m(*args, **kwargs)
        else:
            self.model = efficientnet_v2_l(*args, **kwargs)

    def forward(self, x):
        return self.model(x)
