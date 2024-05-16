import torch
import torch.nn as nn

class VAELoss(nn.Module):
    def __init__(self, λ=1.):
        super().__init__()
        self.λ = λ
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, outputs, target):
        output, vq_loss = outputs
        reconst_loss = self.reconstruction_loss(output, target)

        loss = reconst_loss + self.λ * vq_loss
        return {"loss": loss, "reconstruction loss": reconst_loss, "VQ loss": vq_loss}
