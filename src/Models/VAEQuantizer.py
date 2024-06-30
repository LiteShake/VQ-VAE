
import torch
import torch.nn as nn
import torch.nn.functional as F

#"""
class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.reset_parameters()

    def forward(self, latents):
        #print(latents.shape)
        # Compute L2 distances between latents and embedding weights
        dist = torch.linalg.vector_norm(latents.movedim(1, -1).unsqueeze(-2) - self.embedding.weight, dim=-1)
        encoding_inds = torch.argmin(dist, dim=-1)        # Get the number of the nearest codebook vector
        quantized_latents = self.quantize(encoding_inds)  # Quantize the latents

        # Compute the VQ Losses
        codebook_loss = F.mse_loss(latents.detach(), quantized_latents)
        commitment_loss = F.mse_loss(latents, quantized_latents.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Make the gradient with respect to latents be equal to the gradient with respect to quantized latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return vq_loss, quantized_latents

    def quantize(self, encoding_indices):
        z = self.embedding(encoding_indices)
        z = z.movedim(-1, 1) # Move channels back
        return z

    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)

#"""