import torch
import torch.nn as nn
import torchinfo

from crnet import CRNet
from ncp_fc import NCP_FC
from utils_cnn import BottleneckConvAvePool

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
            ncp_spatial_dim=6,  # reduce data in x-y space before flattened
            ncp_feature_seq=16,  # reduce data in z axes to make the sequence length of 16 (z-axis)
            adaptive_ncp_sensory=None,  # reduce data in x-y axis after flattened (ncp_spatial_dim**2, None to skip)
            **ncp_kwargs,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # remove classifier layers of CRNet
        del self.classifier

        # conv_ave_pool_bottleneck
        self.bottleneck = BottleneckConvAvePool(
            out_spatial=ncp_spatial_dim,  # W x H: 27 x 27 -> 13 x 13
            in_channels=down_features[-1],
            out_channels=ncp_feature_seq  # convolution to change the channels (the RNN sequence length): C: 128 -> 16
        )

        # ncp classifier with data sequence on z-axis (layers)
        layer_features = ncp_spatial_dim ** 2
        if adaptive_ncp_sensory is None:
            self.force_shrink_sensory = False
            adaptive_ncp_sensory = layer_features
        else:  # reduce spatial features if adaptive_ncp_sensory is specified: H*W -> adaptive_ncp_sensory
            self.force_shrink_sensory = True
            self.adaptive_shrink = nn.Linear(ncp_spatial_dim ** 2, adaptive_ncp_sensory)

        # ncp_fc layer
        self.ncp_fc = NCP_FC(seq_len=ncp_feature_seq, classes=classes, bi_directional=bi_directional,
                             sensory_neurons=adaptive_ncp_sensory, **ncp_kwargs)

    def forward(self, x):
        # CRNet
        for down in self.down_samples:
            x = down(x)

        # bottleneck
        x = self.bottleneck(x)

        # Replace FC with NCP_FC
        x = torch.flatten(x, start_dim=2)  # (B, C, H, W) -> (B, C, -1)
        if self.force_shrink_sensory:
            x = self.adaptive_shrink(x)  # reduce features in x-y axes after flattened
        x = self.ncp_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNetNCP_ZRNN(classes=2, in_channels=3, img_dim=224, bi_directional=True, adaptive_ncp_sensory=32)
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNetNCP_ZRNN OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
