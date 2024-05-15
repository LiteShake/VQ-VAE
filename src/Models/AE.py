
import torch
import torch.nn as nn

from AEDecoder import*
from AEEncoder import*

class AE(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.enc = AEEncoder()
        self.dec = AEDecoder()

    def forward(self, x):

        out = self.enc(x)
        print(f"res {out.shape}")
        print(f"res {out}")
        out = self.dec(out)

        return out

tsr = torch.randn(3, 512, 512)

print(tsr.shape)
print(tsr)

model = AE()
print(model)

res = model(tsr)

print(f"final {res.shape}")
