
import torch
import torch.nn as nn

from Models.Components.ResNetBlock import *
from Models.Components.Attention import *
from Models.Components.Sample import *

import torch.nn.functional as F

class AEEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.mha1 = ScaledDotProductAttention(512)
        self.mha2 = ScaledDotProductAttention(256)
        self.mha3 = ScaledDotProductAttention(128)
        self.mha4 = ScaledDotProductAttention(64)
        self.mha5 = ScaledDotProductAttention(32)
        self.mha6 = ScaledDotProductAttention(16)
        self.mha7 = ScaledDotProductAttention(8 )

        self.r1 = ResNet(3, 3)   # 4
        self.r2 = ResNet(3, 3)   # 4
        self.r3 = ResNet(3, 3)   # 4
        self.r4 = ResNet(3, 3)   # 4
        self.r5 = ResNet(3, 3)   # 4
        self.r6 = ResNet(3, 3)   # 4
        self.r7 = ResNet(3, 3)   # 4

        self.pooler = Pool("max2")    # 4
        self.dropper = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input) :

        out = self.mha1(input, input, input)[1]
        # print(out.shape)
        out = self.relu(self.r1(out))
        out = self.dropper(out)
        out = self.pooler(out)

        out = self.mha2(out, out, out)[1]
        out = self.relu(self.r2(out))
        out = self.dropper(out)
        out = self.pooler(out)

        out = self.mha3(out, out, out)[1]
        out = self.relu(self.r3(out))
        out = self.dropper(out)
        out = self.pooler(out)

        out = self.mha4(out, out, out)[1]
        out = self.relu(self.r4(out))
        out = self.dropper(out)
        out = self.pooler(out)

        out = self.mha5(out, out, out)[1]
        out = self.relu(self.r5(out))
        out = self.dropper(out)
        out = self.pooler(out)

        out = self.mha6(out, out, out)[1]
        out = self.relu(self.r6(out))
        out = self.dropper(out)
        out = self.pooler(out)

        out = self.mha7(out, out, out)[1]
        out = self.relu(self.r7(out))
        out = self.dropper(out)
        out = self.pooler(out)

        return out


#"""
tsr = torch.randn(3, 512, 512)

print(tsr.shape)
print(tsr)

model = AEEncoder()
print(model)

res = model(tsr)

print(f"final {res.shape}")
#"""
