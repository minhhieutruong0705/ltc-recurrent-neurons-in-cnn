import torchvision.models as models
import torch.nn as nn
import torch
import torchinfo

from ncp_fc import NCP_FC


class EfficientNet_B4ZNCP64(nn.Module):
    def __init__(self, classes=5, pretrain=True):
        super().__init__()
        self.efficientnet_b4 = models.efficientnet_b4(pretrained=pretrain)
        self.efficientnet_b4.avgpool = nn.AdaptiveAvgPool2d(output_size=8)
        self.efficientnet_b4.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            NCP_FC(
                seq_len=1792,
                classes=classes,
                bi_directional=False,
                sensory_neurons=64,
                inter_neurons=24,
                command_neurons=12,
                motor_neurons=2,
                sensory_outs=12,
                inter_outs=8,
                recurrent_dense=12,
                motor_ins=12
            )
        )

    def forward(self, x):
        x = self.efficientnet_b4.features(x)
        x = self.efficientnet_b4.avgpool(x)
        x = torch.flatten(x, start_dim=2)
        return self.efficientnet_b4.classifier(x)


if __name__ == '__main__':
    image_size = 380
    x = torch.randn(8, 3, image_size, image_size)
    model = EfficientNet_B4ZNCP64()
    y = model(x)
    assert y.size() == (8, 5)
    print("[ASSERTION] EfficientNet_B4ZNCP64 OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
