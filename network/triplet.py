import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnext50_32x4d

import pytorch_lightning as pl

from ..config import *


class TripletNetwork(pl.LightningModule):
    def __init__(self,
                 margin=MARGIN,
                 pretrained=PRETRAINED,
                 freeze=FREEZE,
                 output_dim=OUTPUT_DIM
                 ):
        super(TripletNetwork, self).__init__()
        self.model = resnext50_32x4d(pretrained=pretrained)

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, output_dim)
        )

        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = [self(data) for data in batch[0]]
        loss = self.loss_fn(anchor, positive, negative)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = [self(data) for data in batch[0]]
        loss = self.loss_fn(anchor, positive, negative)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def extract_feature(self, x):
        return self.model(x)