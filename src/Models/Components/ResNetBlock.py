import torch
import torch.nn as nn
import torch.nn.functional as F
from Components.ResConnector import ResConnector

class ResNet(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(ResNet, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.connector = ResConnector( self.block )

    def forward(self, x):

        out = self.connector(x)

        return out

"""
tsr = torch.rand(1, 3, 16, 16)

block = ResNet(3, 3)

out = block(tsr)

print(out.shape)
"""

