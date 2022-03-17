import torch
import torch.nn as nn
import torchinfo

from crnet import CRNet
from ncp_fc import NCP_FC
from utils_cnn import BottleneckConvAvePool

""" 
CRNetNCP_XYRNN considers a 3D tensor as a sequence of data changing by x-axis and y-axis simultaneously.
With the change of data sequence on x-axis, data on y-z axes is considered as the features of the sequence.
Similarly, x-z data is the features of data sequence on y axis.
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


class CRNetNCP_XYRNN(CRNet):
    def __init__(
            self,
            classes=2,
            in_channels=3,
            img_dim=224,
            down_features=[32, 64, 128],
            bi_directional=False,  # combine the prediction on the backward of input sequence
            ncp_spatial_dim=8,  # reduce data in x-y axes to make the sequence length of 8 (y-axis)
            ncp_feature_shrink=4,  # reduce data in z axis to make 32 sensory nodes (x*z = y*z = 8 * 4)
            adaptive_ncp_sensory=None,  # reduce x-z/y-z features (ncp_spatial_dim*ncp_feature_shrink, None to skip)
            **ncp_kwargs,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # remove classifier layers of CRNet
        del self.classifier

        # conv_ave_pool_bottleneck
        self.bottleneck = BottleneckConvAvePool(
            out_spatial=ncp_spatial_dim,  # W x H: 27 x 27 -> 8 x 8 (reduce data in x-y axes)
            in_channels=down_features[-1],
            out_channels=ncp_feature_shrink  # reduce features of z axis: C: 128 -> 4
        )

        # ncp classifier with data sequence on y-axis and x-axis
        xz_yz_features = ncp_spatial_dim * ncp_feature_shrink
        if adaptive_ncp_sensory is None:
            self.is_adaptive_shrink = False
            adaptive_ncp_sensory = xz_yz_features
        else:  # reduce features in x-z and y-z axes if adaptive_ncp_sensory is specified: -> adaptive_ncp_sensory
            self.is_adaptive_shrink = True
            self.adaptive_shrink_x = nn.Sequential(  # for data sequence on x-axis
                nn.Dropout(p=0.5),
                nn.Linear(xz_yz_features, adaptive_ncp_sensory),
                nn.ReLU(inplace=True)
            )
            self.adaptive_shrink_y = nn.Sequential(  # for data sequence on y-axis
                nn.Dropout(p=0.5),
                nn.Linear(xz_yz_features, adaptive_ncp_sensory),
                nn.ReLU(inplace=True)
            )

        # ncp_fc layers
        merge_neurons = ncp_spatial_dim // 2  # output of a ncp layers (data sequence length divided by 2)

        # ncp on x axis
        self.ncp_fc_x = NCP_FC(
            seq_len=ncp_spatial_dim,
            classes=merge_neurons,
            bi_directional=bi_directional,
            sensory_neurons=adaptive_ncp_sensory,
            **ncp_kwargs
        )

        # ncp on y axis
        self.ncp_fc_y = NCP_FC(
            seq_len=ncp_spatial_dim,
            classes=merge_neurons,
            bi_directional=bi_directional,
            sensory_neurons=adaptive_ncp_sensory,
            **ncp_kwargs
        )

        self.ncp_merge_fc = nn.Linear(in_features=merge_neurons * 2, out_features=classes)

    def forward(self, x):
        # CRNet backbone
        for down in self.down_samples:
            x = down(x)

        # bottleneck
        x = self.bottleneck(x)

        # Replace FC with 2 layers of NCP_FC
        x_ncpx = x.permute(0, 3, 2, 1)  # change the tensor from (bn, feats, y, x) to (bn, x, y, feats)
        x_ncpy = x.permute(0, 2, 3, 1)  # change the tensor from (bn, feats, y, x) to (bn, y, x, feats)

        # Flatten data in x-z and y-z dimensions
        x_ncpx = torch.flatten(x_ncpx, start_dim=2)
        x_ncpy = torch.flatten(x_ncpy, start_dim=2)

        # Adaptive sensory
        if self.is_adaptive_shrink:
            x_ncpx = self.adaptive_shrink_x(x_ncpx)
            x_ncpy = self.adaptive_shrink_y(x_ncpy)

        # NCP for data in x-axis sequence
        x_ncpx = self.ncp_fc_x(x_ncpx)
        # NCP for data in y-axis sequence
        x_ncpy = self.ncp_fc_y(x_ncpy)

        x = torch.cat([x_ncpx, x_ncpy], dim=1)
        x = self.ncp_merge_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNetNCP_XYRNN(classes=2, in_channels=3, img_dim=224, bi_directional=True, adaptive_ncp_sensory=32)
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNetNCP_XYRNN OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
