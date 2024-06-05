
import torch
import torchvision
import torchvision.transforms as T
import os
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

class AnimeLoader:

    def __init__(self) -> None:

        #self.address = "./Data/Anime"               # TESTING ADDRESS
        self.address = "./src/Dataloaders/Data/Anime/images/"   # RUN ADDRESS

    def Load(self):

        IMAGE_SZ = 512

        files = os.listdir(self.address)
        print(self.address + files[0])
        samples = [torchvision.io.read_image(self.address + i) for i in files ]

        stats = (.5, .5, .5), (.5, .5, .5)

        transforms = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(IMAGE_SZ),
            T.Resize(IMAGE_SZ),
            T.ToTensor(),
            T.Normalize(*stats)
        ])

        print(samples[0])
        print(samples[0].shape)
        samples = [transforms(i) for i in samples]
        print(f"Loaded {samples} samples")

        return samples

#"""
def main():
    stats = (.5, .5, .5), (.5, .5, .5)
    loader = AnimeLoader()
    data = loader.Load()

    print("LOADED")

if __name__ == "__main__":
    main()
#"""