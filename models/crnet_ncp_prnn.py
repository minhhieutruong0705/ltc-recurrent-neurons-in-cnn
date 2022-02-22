import torch
import torch.nn as nn
from crnet import CRNet
from ncp_fc import NCP_FC

""" 
CRNetNCP_PRNN considers a 3D tensor as a sequence of data changing by patches, 
while considering the corresponding information in x-y-z axes within a patch the features. 
To get the patches, a square image is divided into sub-squares using a grid. 
Every patch must have the same dimensions of x, y, and z. The idea of
considering the patches as the sequence in RNN is borrowed from Qiwei Yin et. al. in the work of 
"CNN and RNN mixed model for image classification", 2018.

In the original work of Neural circuit policies enabling auditable autonomy by Mathias Lechner et. al., Oct 2020, 
sequence length is 16, number of sensory neuron is 32. 

**ncp_kwargs including (default)
    inter_neurons=12,
    command_neurons=6,
    motor_neurons=1,
    sensory_outs=6,
    inter_outs=4,
    recurrent_dense=6,
    motor_ins=6
"""


class CRNetNCP_PRNN(CRNet):
    def __init__(
            self,
            classes=2,
            in_channels=3,
            img_dim=224,
            down_features=[32, 64, 128],
            ncp_patches_per_side=4,  # last tensor is divided into 4*4 patches to make data sequence of 16
            ncp_patch_spatial=4,  # each patch is a 4x4 (pixels) sub-tensor
            ncp_features_shrink=16,  # reduce data in z axis
            ncp_sensory=32,  # must be smaller than ncp_patch_spatial**2*ncp_features_shrink (None to skip FC)
            **ncp_kwargs,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # # default ncp wiring with 128 sensory neurons (ncp_patch_spatial=4 and ncp_features_shrink=8)
        # if len(ncp_kwargs.keys()) == 0:
        #     ncp_kwargs = {
        #         "inter_neurons": 24,
        #         "command_neurons": 12,
        #         "motor_neurons": 2,
        #         "sensory_outs": 12,
        #         "inter_outs": 8,
        #         "recurrent_dense": 12,
        #         "motor_ins": 12
        #     }

        patch_features = ncp_patch_spatial ** 2 * ncp_features_shrink

        # check validity of ncp_sensory
        if ncp_sensory is not None:
            assert ncp_sensory < patch_features
            self.shrink_sensory = True
        else:
            ncp_sensory = patch_features
            self.shrink_sensory = False

        # end global average pooling: W x H: 27 x 27 -> 4*4 x 4*4
        self.global_avg_pool = nn.AdaptiveAvgPool2d(ncp_patches_per_side * ncp_patch_spatial)

        # convolution to change the channels of the last tensor: C: 128 -> 16
        self.feature_shrink_conv = nn.Conv2d(
            in_channels=down_features[-1], out_channels=ncp_features_shrink,
            kernel_size=1, padding=0, stride=1, bias=True
        )

        # reduce features of a patch
        self.patch_shrink = nn.Linear(patch_features, ncp_sensory)  # 4*4*16 -> 32

        # ncp_fc layer
        self.ncp_fc = NCP_FC(
            seq_len=ncp_patch_spatial ** 2,  # number of patches
            classes=classes,
            sensory_neurons=ncp_sensory,  # x-y-z values within a patch
            **ncp_kwargs,
        )

    def forward(self, x):
        # CRNet
        for down in self.downsamples:
            x = down(x)
        x = self.global_avg_pool(x)

        # Replace FC with NCP_FC
        x = self.feature_shrink_conv(x)

        # patching -> flatten -> shrink patch -> ncp


        x = torch.flatten(x, start_dim=2)  # (B, C, H, W) -> (B, C, -1)
        x = self.patch_shrink(x)
        x = self.ncp_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNetNCP_PRNN(classes=2, in_channels=3, img_dim=224)
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNetNCP_ZRNN OK!")
    print(model)
