
import torch
import torch.nn as nn
from Models.VAEQuantizer import *
from Models.AEDecoder import *
import matplotlib.pyplot as plt

class Model(nn.Module):

    def __init__(self, device) -> None:
        super().__init__()

        self.device = device

        self.quantizer = VectorQuantizer(1024, 32, .25)
        self.decoder = AEDecoder()

        self.quantizer.load_state_dict(torch.load("./Saves/Bottleneck/Hikari_M01_VB512_B24.pt"))
        self.decoder.load_state_dict(torch.load("./Saves/Decoder/Hikari_M01_VD512_B24.pt"))

        self.quantizer = self.quantizer.to(device)
        self.decoder = self.decoder.to(device)

    def forward(self):

        with torch.no_grad():

            input = torch.rand(3, 32, 32).to(self.device)

            quants = self.quantizer(input)[1]
            image = self.decoder(quants)

            return image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(device).to(device)
image = model()

print(image)
print(image.shape)

image = image.cpu()

plt.imshow( image.permute(1, 2, 0) )
plt.show()
