from Dataloaders.AnimeLoader import *
from Models.VQVAE import VQVAE
from Trainer import *
import os

def main():
    print(f"CWD {os.getcwd()}")

    loader = AnimeLoader()

    data = loader.Load()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VQVAE(device).to(device)

    trainer = Trainer()
    trainer.Attach(model, data)
    trainer.train(device)

if __name__ == "__main__": main()
