
import torch
import torch.nn as nn

from AEDecoder import*
from AEEncoder import*
from VAEBottleneck import*
from VAEQuantizer import *

class VQVAE(nn.Module):

    def __init__(self, device) -> None:
        super().__init__()

        self.enc = AEEncoder().to(device)
        self.bnk = VectorQuantizer(1024, 32)
        self.dec = AEDecoder().to(device)

    def forward(self, x):

        out = self.enc(x)
        out, vqloss = self.bnk(out)
        # print(out.shape)
        out = self.dec(out)

        return out, vqloss

tsr = torch.randn(3, 512, 512)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(tsr.shape)
print(tsr)
tsr = tsr.to(device)

model = VQVAE(device).to(device)
print(model)

res = model(tsr)

print(f"mean {res[0].shape}")
print(f"vqloss {res[1]}")