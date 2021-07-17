from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from . import trainer
from ..data import TripletsDataset
from ..config import *
from ..model import TripletNetwork

images = ImageFolder()

train_dataset = TripletsDataset()
val_dataset = TripletsDataset()

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=True)
val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                        drop_last=True,
                        shuffle=False)

net = TripletNetwork()

if __name__ == '__main__':
    trainer.fit(net, train_dataloader=train_loader, val_dataloaders=val_loader)
