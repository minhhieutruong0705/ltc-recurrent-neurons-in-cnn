import torch
import torch.nn as nn
from models.crnet import CRNet
from models.ncp_fc import NCP_FC

""" 
CRNetNCP_YRNN considers a 3D tensor as a sequence of data changing by y-axis, 
while considering the corresponding information in x-z axes the features.
feature_enrich_factors of 4 is borrowed from the original implemtation of 
convolutional head in Neural circuit policies enabling auditable autonomy 
by Mathias Lechner et. al., Oct 2020
"""


class CRNetNCP_YRNN(CRNet):
    def __init__(self, classes=2, in_channels=3, img_dim=224, down_features=[32, 64, 128], feature_shrink_value=4):
        super().__init__(classes=classes, in_channels=in_channels,
                         img_dim=img_dim, down_features=down_features)

        # end global average pooling: W x H -> 8 x 8
        last_img_dim = 8
        self.global_avg_pool = nn.AdaptiveAvgPool2d(last_img_dim)

        # reduce features of z axes
        self.feat_shrink = nn.Linear(down_features[-1], feature_shrink_value)

        # ncp_fc layer
        self.ncp_fc = NCP_FC(
            seq_len=last_img_dim,  # changes in y axis
            classes=classes,
            sensory_neurons=last_img_dim * feature_shrink_value,  # x-z values are changing values
        )

    def forward(self, x):
        # CRNet
        for down in self.downsamples:
            x = down(x)
        x = self.global_avg_pool(x)

        # Replace FC with NCP_FC
        # change the tensor from (bn, feats, y, x) to (bn, y, x, feats)
        x = x.permute(0, 2, 3, 1)
        x = self.feat_shrink(x)
        x = torch.flatten(x, start_dim=2)
        x = self.ncp_fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(16, 3, 224, 224)
    model = CRNetNCP_YRNN(classes=2, in_channels=3, img_dim=224)
    y = model(x)
    assert y.size() == (16, 2)
    print("[ASSERTION] CRNetNCP_YRNN OK!")
