
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
        self.address = "./Dataloaders/Data/Anime/images/"   # RUN ADDRESS

    def Load(self):

        IMAGE_SZ = 512

        files = os.listdir(self.address)
        print(self.address + files[0])
        samples = [torchvision.io.read_image(self.address + i) for i in files[:1_280] ]

        stats = (.5, .5, .5), (.5, .5, .5)

        transforms = T.Compose([
            T.ToPILImage(),
            #T.CenterCrop(IMAGE_SZ),
            T.Resize(IMAGE_SZ),
            T.ToTensor(),
            T.Normalize(*stats)
        ])

        samples = [transforms(i) for i in samples]
        for i in samples :
            if(i.shape != (3, 512, 512)) : print('false')
        print(samples[0])
        print(samples[0].shape)
        print(f"Loaded {len(samples)} samples")

        batches = []

        count = 0
        samplecount = 0
        batchcount = 0
        batch = []

        for sample in samples:
            batch.append(sample)
            count += 1
            samplecount += 1

            if(count == 16):
                batches.append(batch)
                batchcount += 1
                count = 0
                batch = []

        print(f"LOADED \n{samplecount} SAMPLES \n{batchcount} BATCHES")

        return batches

"""
def main():
    stats = (.5, .5, .5), (.5, .5, .5)
    loader = AnimeLoader()
    data = loader.Load()

    print("LOADED")

if __name__ == "__main__":
    main()
#"""