import torch
import torch.nn as nn

class VAELoss(nn.Module):
    def __init__(self, λ=1.):
        super().__init__()
        self.λ = λ
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, outputs, vq_loss, target):
        reconst_loss = self.reconstruction_loss(outputs, target)
        #print(reconst_loss)
        loss = reconst_loss + self.λ * vq_loss

        return loss
