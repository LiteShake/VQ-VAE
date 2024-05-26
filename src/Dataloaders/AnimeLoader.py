
import torch
import torchvision
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

class AnimeLoader:

    def __init__(self) -> None:

        #self.address = "./Data/Anime"               # TESTING ADDRESS
        self.address = "./src/Dataloaders/Data/Anime/"   # RUN ADDRESS

    def Load(self):

        IMAGE_SZ = 512
        BATCH_SZ = 128

        stats = (.5, .5, .5), (.5, .5, .5)

        train = ImageFolder(
            self.address,
            transform= T.Compose([
                T.Resize(IMAGE_SZ),
                T.CenterCrop(IMAGE_SZ),
                T.ToTensor(),
                T.Normalize(*stats)
            ])
        )

        train_dl = DataLoader(train, BATCH_SZ, shuffle=True, num_workers=1, pin_memory=True)

        return train_dl

#"""
def main():
    stats = (.5, .5, .5), (.5, .5, .5)
    def denorm(img_tensors):
        return img_tensors * stats[1][0] + stats[0][0]

    def show_images(images, nmax=64):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
        plt.title("[UK-M02] M01 V64 TR")
        plt.show()

    def show_batch(dl, nmax=64):
        for images, _ in dl:
            show_images(images, nmax)
            break

    loader = AnimeLoader()
    data = loader.Load()

    print("LOADED")
    num = 0
    for _, dat in enumerate(data, 0):
        print( dat[0].shape )

    print(f"batches {num}")

    show_batch(data)

    print("VIZ")

if __name__ == "__main__":
    main()
#"""