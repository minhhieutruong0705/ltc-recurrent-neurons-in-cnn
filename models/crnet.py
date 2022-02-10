import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=not batch_norm,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return self.maxpool(x)


class CRNet(nn.Module):
    def __init__(self, classes=2, in_channels=3, img_dim=224, down_features=[32, 64, 128]):
        super().__init__()

        # feature extraction
        self.downsamples = nn.ModuleList()
        for feature in down_features:
            self.downsamples.append(
                ConvPool(
                    in_channels=in_channels,
                    out_channels=feature,
                    batch_norm=(
                        feature == 32
                    ),  # batch normalization is applied to first convolutional block
                )
            )
            in_channels = feature
            img_dim = (img_dim - 1) // 2

        # end global average pooling
        self.global_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        img_dim = (img_dim - 1) // 2

        # fully connected classifier
        self.head_img_dim = img_dim
        self.head_channels = in_channels
        self.classifier = nn.Linear(self.head_img_dim ** 2 * self.head_channels, classes)

    def forward(self, x):
        for down in self.downsamples:
            x = down(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == "__main__":
    image_size = 224
    x = torch.randn(8, 3, image_size, image_size)
    model = ConvPool(in_channels=3, out_channels=32, batch_norm=True)
    y = model(x)
    assert y.size() == (8, 32, (image_size - 1) // 2, (image_size - 1) // 2)
    print("[ASSERTION] ConvPool OK!")


if __name__ == "__main__":
    image_size = 224
    x = torch.randn(8, 3, image_size, image_size)
    model = CRNet(classes=2, in_channels=3, img_dim=224)
    y = model(x)
    y = F.log_softmax(y, dim=1)
    assert y.size() == (8, 2)
    print("[ASSERTION] CRNet OK!")
