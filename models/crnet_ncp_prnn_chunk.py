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
            seq_horizontal=True,  # patches in sequence of rows or columns
            seq_zigzag=False,  # queuing order or zigzag order
            ncp_patches_per_side=4,  # last tensor is divided into 4*4 patches to make data sequence of 16
            ncp_patch_spatial=2,  # each patch is a 2x2 (pixels) sub-tensor
            ncp_features_shrink=8,  # reduce data in z axis
            adaptive_ncp_sensory=None,  # reduce patch features (ncp_patch_spatial**2*ncp_features_shrink, None to skip)
            **ncp_kwargs,
    ):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # remove classifier layers of CRNet
        del self.classifier

        # conv_ave_pool_bottleneck
        self.bottleneck = BottleneckConvAvePool(
            out_spatial=ncp_patches_per_side * ncp_patch_spatial,  # W x H: 27 x 27 -> 4*2 x 4*2
            in_channels=down_features[-1],
            out_channels=ncp_features_shrink  # reduce features of z axis: C: 128 -> 8
        )

        # non-overlapping patching using chunker
        self.chunker = Chunker(chunks_per_side=ncp_patches_per_side, horizontal_seq=seq_horizontal, zigzag=seq_zigzag)

        # ncp classifier with data sequence on patches
        patch_features = ncp_patch_spatial ** 2 * ncp_features_shrink
        if adaptive_ncp_sensory is None:
            self.is_adaptive_shrink = False
            adaptive_ncp_sensory = patch_features
        else:  # reduce features of a patch if adaptive_ncp_sensory is specified: H_p*W_p*C -> adaptive_ncp_sensory
            self.is_adaptive_shrink = True
            self.adaptive_shrink = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(patch_features, adaptive_ncp_sensory),
                nn.ReLU(inplace=True)
            )

        # ncp_fc layer
        self.ncp_fc = NCP_FC(seq_len=ncp_patches_per_side ** 2, classes=classes, bi_directional=bi_directional,
                             sensory_neurons=adaptive_ncp_sensory, **ncp_kwargs)

    def forward(self, x):
        # CRNet backbone
        for down in self.down_samples:
            x = down(x)

        # bottleneck
        x = self.bottleneck(x)

        # Replace FC with NCP_FC
        x = self.chunker(x)  # non-overlapping patching
        x = torch.flatten(x, start_dim=2)  # (B, P, C, H_p, W_p) -> (B, P, -1); features of a patch is C*H_p*W_p
        if self.is_adaptive_shrink:
            x = self.adaptive_shrink(x)
        x = self.ncp_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(8, 3, 224, 224)
    model = CRNetNCP_ChunkPRNN(classes=2, in_channels=3, img_dim=224, bi_directional=True, adaptive_ncp_sensory=32)
    y = model(x)
    assert y.size() == (8, 2)
    print("[ASSERTION] CRNetNCP_ChunkPRNN OK!")
    print(model)
    torchinfo.summary(model=model, input_data=x, device="cpu")
