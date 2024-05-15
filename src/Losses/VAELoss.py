
import torch
import torch.nn as nn

def VAELoss(x, xhat, mean, log_var):

    red_loss = nn.functional.binary_cross_entropy(xhat, x, reduction="sum")
    KLD = -.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp() )

    return red_loss + KLD

"""
mean = torch.rand(3, 32, 32)
logv = torch.rand(3, 32, 32)
out = torch.rand(3, 512, 512)
xhat = torch.rand(3, 512, 512)

loss = VAELoss(out, xhat, mean, logv)
print(loss)
"""
