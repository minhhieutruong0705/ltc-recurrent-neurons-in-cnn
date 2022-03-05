import torch
import torch.nn as nn
from crnet import CRNet
from ncp_fc import NCP_FC

""" 
CRNetNCP_ZRNN considers a 3D tensor as a sequence of data changing by z-axis, 
while considering the corresponding information in x-y axes the features. In the original work of
Neural circuit policies enabling auditable autonomy by Mathias Lechner et. al., Oct 2020, sequence length
is 16, number of sensory neuron is 32. To be complement with the changing in z-axis, number of channels
is set 16, and number of x-y values is shrink to 32 (default)

**ncp_kwargs including (default)
    inter_neurons=12,
    command_neurons=6,
    motor_neurons=1,
    sensory_outs=6,
    inter_outs=4,
    recurrent_dense=6,
    motor_ins=6
"""


class CRNetNCP_ZRNN(CRNet):
    def __init__(
            self,
            classes=2,
            in_channels=3,
            img_dim=224,
            down_features=[32, 64, 128],
            bi_directional=False,  # combine the prediction on the backward of input sequence
            ncp_feature=16,  # reduce data in z axes to make the sequence length of 16 (z-axis)
            ncp_spatial_shrink=32,  # reduce data in x-y axis to make 32 sensory nodes (after flattened)
            **ncp_kwargs,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # convolution to change the channels (the RNN sequence length): C: 128 -> 16
        self.ncp_seq_conv = nn.Conv2d(
            in_channels=down_features[-1], out_channels=ncp_feature,
            kernel_size=1, padding=0, stride=1, bias=True
        )

        # reduce features of x-y axes (x = y i.e. square image)
        self.feat_shrink = nn.Linear(self.head_img_dim ** 2, ncp_spatial_shrink)  # 13*13 -> 32

        # ncp_fc layer
        self.ncp_fc = NCP_FC(seq_len=ncp_feature, classes=classes, bi_directional=bi_directional,
                             sensory_neurons=ncp_spatial_shrink, **ncp_kwargs)

    def forward(self, x):
        # CRNet
        for down in self.downsamples:
            x = down(x)
        x = self.global_avg_pool(x)

        # Replace FC with NCP_FC
        x = self.ncp_seq_conv(x)
        x = torch.flatten(x, start_dim=2)  # (B, C, H, W) -> (B, C, -1)
        x = self.feat_shrink(x)  # reduce features in x-y axes
        x = self.ncp_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNetNCP_ZRNN(classes=2, in_channels=3, img_dim=224, bi_directional=True)
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNetNCP_ZRNN OK!")
    print(model)
