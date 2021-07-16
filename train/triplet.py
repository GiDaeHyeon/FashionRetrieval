import torch
from model import TripletModel

import pytorch_lightning as pl

from config import *


class TripletNetwork(pl.LightningModule):
    def __init__(self,
                 margin=MARGIN):
        super(TripletNetwork, self).__init__()
        self.model = TripletModel()
        self.loss_fn = torch.nn.TripletMarginLoss(margin, )

    def forward(self, batch):
        output = self.model([*batch])
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, last_epoch=-1, verbose=True)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = self(batch[0])
        loss = self.loss_fn(anchor, positive, negative)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = self(batch[0])
        loss = self.loss_fn(anchor, positive, negative)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def extract_feature(self, x):
        feature = self.model(x[None])
        return feature