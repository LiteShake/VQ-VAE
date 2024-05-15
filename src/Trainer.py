
import torch
import torch.nn as nn
from Losses.VAELoss import VAELoss

class Trainer :

    def __init__(self) -> None:
        pass

    def Attach(self, model, data) :
        self.model = model
        self.data = data

    def train(self, device, hypers):

        optimizer = torch.optim.AdamW(hypers["LEARNRATE"])

        for batch in self.data:

            # loop over the dataset multiple times
            for epoch in range( hypers['EPOCHS'] ):
                running_loss = 0.0
                for image in batch :
                    img, recon = image, image

                    img, recon = img.to(device), recon.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(img)
                    loss = VAELoss(img, recon, mean, logvar)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                print('Loss: {}'.format(running_loss))

            print('Finished Training')