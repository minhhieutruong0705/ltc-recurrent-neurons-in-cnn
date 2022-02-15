import torch
import torch.nn as nn
from crnet import CRNet

""" 
CRNet_3FC customs the original CRNet with a more complex classification layer.
The layer consists of 3 fully connected blocks reducing the size
from 13*13*128 to 1000 to 100 and finally number of classes. The implementation
is borrowed from the original implementation of the "feedforward CNN" in
Neural circuit policies enabling auditable autonomy by Mathias Lechner et. al., Oct 2020 
"""


class CRNet_3FC(CRNet):
    def __init__(self, classes=2, in_channels=3, img_dim=224, down_features=[32, 64, 128]):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        self.classifier = nn.Sequential(
            nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=self.head_img_dim ** 2 * self.head_channels, out_features=1000),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=1000, out_features=100),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features=100, out_features=classes)
            )
        )

    def forward(self, x):
        for down in self.downsamples:
            x = down(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNet_3FC(classes=2, in_channels=3, img_dim=224)
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNet_3FC OK!")
