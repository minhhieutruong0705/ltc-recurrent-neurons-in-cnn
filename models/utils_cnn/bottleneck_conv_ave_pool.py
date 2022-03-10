import torch
import torch.nn as nn


class BottleneckConvAvePool(nn.Module):
    def __init__(self, out_spatial, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True
        )  # reduce features of z axis
        self.ave_pooling = nn.AdaptiveAvgPool2d(out_spatial)  # reduce data in x-y axes

    def forward(self, x):
        x = self.conv(x)
        return self.ave_pooling(x)


if __name__ == '__main__':
    image_size = 27
    x = torch.randn(8, 128, image_size, image_size)
    model = BottleneckConvAvePool(out_spatial=8, in_channels=128, out_channels=16)
    y = model(x)
    assert y.size() == (8, 16, 8, 8)
    print("[ASSERTION] BottleneckConvAvePool OK!")
    print(model)
