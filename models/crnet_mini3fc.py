import torch
import torch.nn as nn
import torchinfo

from crnet import CRNet
from utils_cnn import BottleneckConvAvePool

""" 
CRNet_Mini3FC customs the original CRNet_3FC with a more flexible bottle layer.
The layer allows the control of channels and spatial dimensions of tensors 
at the end of down_samples. The operation helps the reduction of flattened parameters
before feeding into fully-connected layers. Moreover, CRNet_Mini3FC also provide the
modification of the fully-connected layers.
"""


class CRNet_Mini3FC(CRNet):
    def __init__(
            self,
            classes=2,
            in_channels=3,
            img_dim=224,
            down_features=[32, 64, 128],
            bottleneck_spatial=13,  # similar to original CRNet and CRNet_3FC
            bottleneck_channels=128,  # similar to original CRNet and CRNet_3FC
            fc1_channels=1000,
            fc2_channels=100,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        self.bottleneck = BottleneckConvAvePool(
            out_spatial=bottleneck_spatial,
            in_channels=down_features[-1],
            out_channels=bottleneck_channels
        )

        self.classifier = nn.Sequential(
            nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=bottleneck_spatial ** 2 * bottleneck_channels, out_features=fc1_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=fc1_channels, out_features=fc2_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features=fc2_channels, out_features=classes)
            )
        )

    def forward(self, x):
        for down in self.down_samples:
            x = down(x)
        x = self.bottleneck(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNet_Mini3FC(
        classes=2, in_channels=3, img_dim=224,
        bottleneck_spatial=2, bottleneck_channels=8,
        fc1_channels=12, fc2_channels=6
    )
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNet_Mini3FC OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
