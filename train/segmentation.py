import torch.multiprocessing as mp
from model import SegmentationModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from config import *


class TripletNetwork(pl.LightningModule):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.model = customModel()
        self.loss_fn = TripletLoss()
        self.best_loss = 9999
        self.val_loss = 0

    def forward(self, a, p, n):
        embedded_a = self.model(a)
        embedded_p = self.model(p)
        embedded_n = self.model(n)
        return embedded_a, embedded_p, embedded_n

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, last_epoch=-1, verbose=True)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = [*batch[0]]
        anchor, positive, negative = self(anchor, positive, negative)

        loss = self.loss_fn(anchor, positive, negative)

        self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = [*batch[0]]
        anchor, positive, negative = self(anchor, positive, negative)

        loss = self.loss_fn(anchor, positive, negative)

        self.val_loss += loss

        self.log('val_loss', loss)

        return {'val_loss': loss}


    def extract_feature(self, x):
        feature = self.model(x[None])
        return feature