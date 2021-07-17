from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

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

logger = TensorBoardLogger("segmentation log", name=VERSION, default_hp_metric=False)
checkpoint_callback = ModelCheckpoint(
                                      monitor='val_loss',
                                      dirpath=CKPT_DIR,
                                      filename=CKPT_NAME
                                     )
early_stop_callback = EarlyStopping(
                                    monitor='val_loss',
                                    min_delta=0.0001,
                                    patience=10,
                                    verbose=True,
                                    mode='min'
                                    )

trainer = pl.Trainer(max_epochs=MAX_EPOCH,
                     logger=logger,
                     num_sanity_val_steps=0,
                     accelerator='ddp',
                     gpus=GPUS,
                     callbacks=[early_stop_callback,
                                checkpoint_callback])

if __name__ == '__main__':
    trainer.fit(net, train_dataloader=train_loader, val_dataloaders=val_loader)
