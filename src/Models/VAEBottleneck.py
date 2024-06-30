
import torch
import torch.nn as nn

class VAEBottleNeck(nn.Module):

    def __init__(self, device) -> None:
        super().__init__()

        self.mean = nn.Linear(4, 4)#.to(device)
        self.logvar = nn.Linear(4, 4)#.to(device)
        self.device = device

    def forward(self, x):

        x = x.to(self.device)
        # KLD
        mean, logvar = self.mean(x), self.logvar(x)

        # Reparameterization
        eps = torch.randn_like(logvar).to(self.device)
        z = mean + logvar * eps

        return z, mean, logvar

"""
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

tsr = torch.randn(3, 32, 32)

print(tsr.shape)
print(tsr)

model = VAEBottleNeck(device).to(device)
print(model)

res = model(tsr)

print(f"outs {res[0].shape}")
print(f"mean {res[1].shape}")
print(f"logv {res[2].shape}")
"""
