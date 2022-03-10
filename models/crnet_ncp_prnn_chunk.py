import torch
import torch.nn as nn
import torchinfo

from crnet import CRNet
from ncp_fc import NCP_FC
from utils_cnn import BottleneckConvAvePool
from utils_tensor_functional import Chunker

""" 
CRNetNCP_ChunkPRNN considers a 3D tensor as a sequence of data changing by patches, 
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


class CRNetNCP_ChunkPRNN(CRNet):
    def __init__(
            self,
            classes=2,
            in_channels=3,
            img_dim=224,
            down_features=[32, 64, 128],
            bi_directional=False,  # combine the prediction on the backward of input sequence
            ncp_patches_per_side=4,  # last tensor is divided into 4*4 patches to make data sequence of 16
            ncp_patch_spatial=4,  # each patch is a 4x4 (pixels) sub-tensor
            ncp_features_shrink=16,  # reduce data in z axis
            seq_horizontal=True,  # patches in sequence of rows or columns
            seq_zigzag=False,  # queuing order or zigzag order
            force_ncp_sensory=32,  # must be smaller than ncp_patch_spatial**2*ncp_features_shrink (None to skip)
            **ncp_kwargs,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # remove global_avg_pool and classifier layers of CRNet
        del self.global_avg_pool, self.classifier

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

        # check validity of force_ncp_sensory
        if force_ncp_sensory is not None:
            assert force_ncp_sensory < patch_features
            self.force_shrink_sensory = True
        else:
            force_ncp_sensory = patch_features
            self.force_shrink_sensory = False

        # # end global average pooling: W x H: 27 x 27 -> 4*4 x 4*4
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(ncp_patches_per_side * ncp_patch_spatial)

        # # convolution to change the channels of the last tensor: C: 128 -> 16
        # self.feature_shrink_conv = nn.Conv2d(
        #     in_channels=down_features[-1], out_channels=ncp_features_shrink,
        #     kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True
        # )

        # conv_ave_pool_bottleneck
        self.bottleneck = BottleneckConvAvePool(
            out_spatial=ncp_patches_per_side * ncp_patch_spatial,  # W x H: 27 x 27 -> 4*4 x 4*4
            in_channels=down_features[-1],
            out_channels=ncp_features_shrink  # reduce features of z axis: C: 128 -> 16
        )

        # non-overlapping patching using chunker
        self.chunker = Chunker(chunks_per_side=ncp_patches_per_side, horizontal_seq=seq_horizontal, zigzag=seq_zigzag)

        # reduce features of a patch if force_ncp_sensory is specified: patch_features: 4*4*16 -> 32
        self.patch_shrink = nn.Linear(patch_features, force_ncp_sensory) if self.force_shrink_sensory else None

        # ncp_fc classifier layer
        self.ncp_fc = NCP_FC(seq_len=ncp_patches_per_side ** 2, classes=classes, bi_directional=bi_directional,
                             sensory_neurons=force_ncp_sensory, **ncp_kwargs)

    def forward(self, x):
        # CRNet
        for down in self.down_samples:
            x = down(x)

        # bottleneck
        x = self.bottleneck(x)

        # Replace FC with NCP_FC
        x = self.chunker(x)  # non-overlapping patching
        x = torch.flatten(x, start_dim=2)  # (B, P, C, H_p, W_p) -> (B, P, -1); features of a patch is C*H_p*W_p
        if self.force_shrink_sensory:
            x = self.patch_shrink(x)
        x = self.ncp_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(8, 3, 224, 224)
    model = CRNetNCP_ChunkPRNN(classes=2, in_channels=3, img_dim=224, bi_directional=True)
    y = model(x)
    assert y.size() == (8, 2)
    print("[ASSERTION] CRNetNCP_ChunkPRNN OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
