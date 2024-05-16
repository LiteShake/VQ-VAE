
import torch
import torchvision
import torchvision.transforms as T
import os

class AnimeLoader:

    def __init__(self) -> None:

        self.address = "./src/Dataloaders/Data/images"

    def Load(self):

        lst = os.listdir(self.address)
        print(len(lst))
