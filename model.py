import torch
from torch import nn, optim

from torchvision.models import resnext50_32x4d

import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from config import *


###################
# Image Segmentatin
###################

class SegmentationModel(nn.Module):
    def __init__(self,
                 # segmentation model config
                 seg_encoder=SEG_ENCODER,
                 seg_encoder_depth=SEG_ENCODER_DEPTH,
                 seg_encoder_weight=SEG_ENCODER_WEIGHT,
                 classes_num=CLASS_NUM
                 ):
        super(SegmentationModel, self).__init__()
        self.segmentation_model = smp.DeepLabV3Plus(
            encoder_name=seg_encoder,
            encoder_depth=seg_encoder_depth,
            encoder_weights=seg_encoder_weight,
            classes=classes_num
        )

    def forward(self, x):
        return self.segmentation_model(x)


class SegmentationNetwork(pl.LightningModule):
    def __init__(self):
        super(SegmentationNetwork, self).__init__()
        self.model = SegmentationModel()
        self.loss_fn = smp.losses.SoftCrossEntropyLoss()

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, last_epoch=-1, verbose=True)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image, mask = batch
        output = self(image)
        loss = self.loss_fn(mask, output)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        output = self(image)
        loss = self.loss_fn(mask, output)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def get_mask(self, x):
        return self(x)


###################
# Metric Learning
###################

class TripletModel(nn.Module):
    def __init__(self,
                 # classification model config
                 pretrained=PRETRAINED,
                 freeze=FREEZE,
                 output_dim=OUTPUT_DIM
                 ):
        super(TripletModel, self).__init__()
        self.cnn_model = resnext50_32x4d(pretrained=pretrained)

        if freeze:
            for parameter in self.cnn_model.parameters():
                parameter.requires_grad = False

        in_features = self.cnn_model.fc.in_features

        self.cnn_model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.cnn_model(x)


class TripletNetwork(pl.LightningModule):
    def __init__(self,
                 margin=MARGIN):
        super(TripletNetwork, self).__init__()
        self.model = TripletModel()
        self.loss_fn = torch.nn.TripletMarginLoss(margin, )

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, last_epoch=-1, verbose=True)
        return [optimizer], [scheduler]

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
