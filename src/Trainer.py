
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

            print(f"{len(data[0])} BATCHES")
            print(data[0][1])

            for batch in range(len(self.data)):

                for epoch in range(5):

                    print(type(data))
                    print(f"{len(data)} SAMPLES")
                    for sample in self.data[batch]:


                        running_loss = 0.0
                        inputs, labels = sample, sample
                        print(sample.shape)
                        inputs, labels = inputs.to(device), labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)["loss"]
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                print('Loss: {}'.format(running_loss))
                torch.save(self.model.state_dict(), f"./Saves/Hikari_M01_V512_B{batch}.pt")

        print('Finished Training')