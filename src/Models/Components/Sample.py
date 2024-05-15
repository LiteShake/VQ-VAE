import torch
import torch.nn as nn

class Upsample(nn.Module) :

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 4,

        self.up = nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0)

    def forward(self, input):

        upsampled = self.up(input)

        return upsampled

class Pool(nn.Module) :

    def __init__(self, type) -> None:
        super().__init__()

        match(type):

            case "max" : self.sample = nn.MaxPool1d(2)
            case "avg" : self.sample = nn.AvgPool1d(2)

            case "max2" : self.sample = nn.MaxPool2d(2)
            case "avg2" : self.sample = nn.AvgPool2d(2)

    def forward(self, input):

        sampled = self.sample(input)

        return sampled

"""
tsr = torch.rand(1, 3, 4, 4)

block = Upsample()

out = block(tsr)

print(out.shape)
"""
