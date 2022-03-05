import torch
import torch.nn as nn
from crnet import CRNet
from ncp_fc import NCP_FC

""" 
CRNetNCP_YRNN considers a 3D tensor as a sequence of data changing by y-axis, 
while considering the corresponding information in x-z axes the features.
ncp_feature_shrink value of 4 is borrowed from the "features_per_layer" variable
in the original implementation of convolutional head in 
Neural circuit policies enabling auditable autonomy by Mathias Lechner et. al., Oct 2020

**ncp_kwargs including (default)
    inter_neurons=12,
    command_neurons=6,
    motor_neurons=1,
    sensory_outs=6,
    inter_outs=4,
    recurrent_dense=6,
    motor_ins=6
"""


class CRNetNCP_YRNN(CRNet):
    def __init__(
            self,
            classes=2,
            in_channels=3,
            img_dim=224,
            down_features=[32, 64, 128],
            bi_directional=False,  # combine the prediction on the backward of input sequence
            ncp_spatial_dim=8,  # reduce data in x-y axes to make the sequence length of 8 (y-axis)
            ncp_feature_shrink=4,  # reduce data in z axis to make 32 sensory nodes (x*z = 8 * 4)
            **ncp_kwargs,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # end global average pooling: W x H: 27 x 27 -> 8 x 8
        self.global_avg_pool = nn.AdaptiveAvgPool2d(ncp_spatial_dim)

        # reduce features of z axes
        self.feat_shrink = nn.Linear(down_features[-1], ncp_feature_shrink)

        # ncp_fc layer
        self.ncp_fc = NCP_FC(seq_len=ncp_spatial_dim, classes=classes, bi_directional=bi_directional,
                             sensory_neurons=ncp_spatial_dim * ncp_feature_shrink, **ncp_kwargs)

    def forward(self, x):
        # CRNet
        for down in self.downsamples:
            x = down(x)
        x = self.global_avg_pool(x)

        # Replace FC with NCP_FC
        x = x.permute(0, 2, 3, 1)  # change the tensor from (bn, feats, y, x) to (bn, y, x, feats)
        x = self.feat_shrink(x)  # shrink features in z-axis
        x = torch.flatten(x, start_dim=2)
        x = self.ncp_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNetNCP_YRNN(classes=2, in_channels=3, img_dim=224, bi_directional=True)
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNetNCP_YRNN OK!")
    print(model)
