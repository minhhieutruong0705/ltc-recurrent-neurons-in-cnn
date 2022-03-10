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
            ncp_spatial_dim=13,  # reduce data in x-y space before flattened
            ncp_feature_seq=16,  # reduce data in z axes to make the sequence length of 16 (z-axis)
            force_ncp_sensory=32,  # reduce data in x-y axis to make 32 sensory nodes (after flattened, None to skip)
            **ncp_kwargs,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # remove global_avg_pool and classifier layers of CRNet
        del self.global_avg_pool, self.classifier

        layer_features = ncp_spatial_dim ** 2

        # check validity of force_ncp_sensory
        if force_ncp_sensory is not None:
            assert force_ncp_sensory < layer_features
            self.force_shrink_sensory = True
        else:
            force_ncp_sensory = layer_features
            self.force_shrink_sensory = False

        # # convolution to change the channels (the RNN sequence length): C: 128 -> 16
        # self.ncp_seq_conv = nn.Conv2d(
        #     in_channels=down_features[-1], out_channels=ncp_feature,
        #     kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True
        # )

        # conv_ave_pool_bottleneck
        self.bottleneck = BottleneckConvAvePool(
            out_spatial=ncp_spatial_dim,  # W x H: 27 x 27 -> 13 x 13
            in_channels=down_features[-1],
            out_channels=ncp_feature_seq  # convolution to change the channels (the RNN sequence length): C: 128 -> 16
        )

        # reduce features of x-y axes (x = y i.e. square image) if force_ncp_sensory is specified: H*W: 13*13 -> 32
        # Note: This linear operation could be replaced by an adaptive pooling operation
        # However, this operation is better as it has trainable parameters
        self.feat_shrink = nn.Linear(ncp_spatial_dim ** 2, force_ncp_sensory) if self.force_shrink_sensory else None

        # ncp_fc classifier layer
        self.ncp_fc = NCP_FC(seq_len=ncp_feature_seq, classes=classes, bi_directional=bi_directional,
                             sensory_neurons=force_ncp_sensory, **ncp_kwargs)

    def forward(self, x):
        # CRNet
        for down in self.down_samples:
            x = down(x)

        # bottleneck
        x = self.bottleneck(x)

        # Replace FC with NCP_FC
        x = torch.flatten(x, start_dim=2)  # (B, C, H, W) -> (B, C, -1)
        if self.force_shrink_sensory:
            x = self.feat_shrink(x)  # reduce features in x-y axes after flattened
        x = self.ncp_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNetNCP_ZRNN(classes=2, in_channels=3, img_dim=224, bi_directional=True)
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNetNCP_ZRNN OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
