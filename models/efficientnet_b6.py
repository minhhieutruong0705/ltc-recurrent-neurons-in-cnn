import torchvision.models as models
import torch.nn as nn
import torch
import torchinfo


class EfficientNet_B6(nn.Module):
    def __init__(self, classes=5, pretrain=True):
        super().__init__()
        self.efficientnet_b6 = models.efficientnet_b6(pretrained=pretrain)
        self.efficientnet_b6.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2304, out_features=1024),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1024, out_features=classes)
        )

    def forward(self, x):
        return self.efficientnet_b6(x)


if __name__ == '__main__':
    image_size = 528
    x = torch.randn(8, 3, image_size, image_size)
    model = EfficientNet_B6()
    y = model(x)
    assert y.size() == (8, 5)
    print("[ASSERTION] EfficientNet_B6 OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
