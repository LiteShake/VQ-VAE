
import torch
import torch.nn as nn

from Models.Components.ResNetBlock import *
from Models.Components.Attention import *
from Models.Components.Sample import *

import torch.nn.functional as F

class AEDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.mha1 = ScaledDotProductAttention(4 )
        self.mha2 = ScaledDotProductAttention(8 )
        self.mha3 = ScaledDotProductAttention(16)
        self.mha4 = ScaledDotProductAttention(32)
        self.mha5 = ScaledDotProductAttention(64)
        self.mha6 = ScaledDotProductAttention(128)

        self.r1 = ResNet(3, 3)   # 4
        self.r2 = ResNet(3, 3)   # 4
        self.r3 = ResNet(3, 3)   # 4
        self.r4 = ResNet(3, 3)   # 4
        self.r5 = ResNet(3, 3)   # 4
        self.r6 = ResNet(3, 3)   # 4

        self.upsample = Upsample()    # 4
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input) :

        out = self.mha1(input, input, input)[1]
        out = self.relu(self.r1(out))
        #print(out.shape)
        out = self.upsample(out)
        # print(f"upsample {out.shape}")

        out = self.mha2(input, input, input)[1]
        out = self.relu(self.r2(out))
        #print(out.shape)
        out = self.upsample(out)
        # print(f"upsample {out.shape}")

        out = self.mha3(out, out, out)[1]
        out = self.relu(self.r3(out))
        out = self.upsample(out)

        out = self.mha4(out, out, out)[1]
        out = self.relu(self.r4(out))
        out = self.upsample(out)

        out = self.mha5(out, out, out)[1]
        out = self.relu(self.r5(out))
        out = self.upsample(out)

        out = self.mha6(out, out, out)[1]
        out = self.sigmoid(self.r6(out))

        return out
