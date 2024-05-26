
import torch
import torch.nn as nn

from Models.AEDecoder import*
from Models.AEEncoder import*
from Models.VAEBottleneck import*

class VAE(nn.Module):

    def __init__(self, device) -> None:
        super().__init__()

        self.enc = AEEncoder().to(device)
        self.bnk = VAEBottleNeck(device).to(device)
        self.dec = AEDecoder().to(device)

    def forward(self, x):

        out = self.enc(x)
        out, mean, logvar = self.bnk(out)
        print(out.shape)
        out = self.dec(out)

        return mean, logvar, out

"""
tsr = torch.randn(3, 512, 512)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(tsr.shape)
print(tsr)
tsr = tsr.to(device)

model = VAE(device).to(device)
print(model)

res = model(tsr)

print(f"mean {res[0].shape}")
print(f"logv {res[1].shape}")
print(f"outs {res[2].shape}")
"""
