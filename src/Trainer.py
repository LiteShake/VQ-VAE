
import torch
import torch.nn as nn
from Losses.VQVAELoss import VAELoss

"""
# loop over the dataset multiple times
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Loss: {}'.format(running_loss)

print('Finished Training')
"""

class Trainer :

    def __init__(self) -> None:
        pass

    def Attach(self, data) :
        self.data = data

    def train(self, model, device):

        optimizer = torch.optim.AdamW(model.parameters())
        criterion = VAELoss()

        # loop over the dataset multiple times

        skippedsamples = 0
        for batch in range(len(self.data)):

            for epoch in range(64):

                batch_loss = 0.0

                for sample in self.data[batch]:

                    #print(f"SAMPLE {sample.shape}")

                    try :
                        inputs, labels = sample, sample
                        #print("INPUTS ",inputs.shape)
                        inputs, labels = inputs.to(device), labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = model(inputs)
                        loss = criterion(outputs[0], outputs[1], labels)
                        loss.backward()
                        optimizer.step()

                        batch_loss += loss.item()

                    except :
                        skippedsamples += 1
                        print(f"Samples Skipped | {skippedsamples}")
                        batch_loss += loss.item()

                print('E: {} BL: {}'.format(epoch, batch_loss / 16))
            torch.save(model.enc.state_dict(), f"./Saves/Encoder/Hikari_M01_VE512_B{batch}.pt")
            torch.save(model.dec.state_dict(), f"./Saves/Decoder/Hikari_M01_VD512_B{batch}.pt")
            torch.save(model.bnk.state_dict(), f"./Saves/Bottleneck/Hikari_M01_VB512_B{batch}.pt")

        print('Finished Training')