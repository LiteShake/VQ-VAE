
import torch
import torch.nn as nn
from Losses.VQVAELoss import VAELoss

class Trainer :

    def __init__(self) -> None:
        pass

    def Attach(self, model, data) :
        self.model = model
        self.data = data

    def train(self, device):

        optimizer = torch.optim.AdamW(self.model.parameters())
        criterion = VAELoss()
        loader = self.data

        # loop over the dataset multiple times
        for _, data in enumerate(loader, 0):

            print(f"{len(data)} {len(data[0])} {data[0][0].shape}")
            #data = data[0]

            print(data[0].shape)
            skippedsamples = 0

            for batch in range(len(data[0])):

                for epoch in range(5):

                    print(type(data))
                    print(f"{data[batch][0].shape} SAMPLES")

                    for sample in data[batch]:

                        #print(f"SAMPLE {sample.shape}")

                        try :
                            running_loss = 0.0
                            inputs, labels = sample, sample
                            #print("INPUTS ",inputs.shape)
                            inputs, labels = inputs.to(device), labels.to(device)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward + backward + optimize
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)["loss"]
                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item()

                        except :
                            skippedsamples += 1
                            print(f"Samples Skipped | {skippedsamples}")
                            running_loss += loss.item()

                    print('Batch Loss: {}'.format(running_loss))
                torch.save(self.model.enc.state_dict(), f"./src/Saves/Hikari_M01_VE512_B{batch}.pt")
                torch.save(self.model.dec.state_dict(), f"./src/Saves/Hikari_M01_VD512_B{batch}.pt")
                torch.save(self.model.bnk.state_dict(), f"./src/Saves/Hikari_M01_VB512_B{batch}.pt")

        print('Finished Training')