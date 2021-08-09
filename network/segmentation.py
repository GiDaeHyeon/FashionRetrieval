import torch
from torch import optim

import segmentation_models_pytorch as smp
import pytorch_lightning as pl

import torchmetrics

from ..config import *


class SegmentationNetwork(pl.LightningModule):
    def __init__(self,
                 seg_encoder=SEG_ENCODER,
                 seg_encoder_depth=SEG_ENCODER_DEPTH,
                 seg_encoder_weight=SEG_ENCODER_WEIGHT,
                 classes_num=CLASS_NUM
                 ):
        super(SegmentationNetwork, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=seg_encoder,
            encoder_depth=seg_encoder_depth,
            encoder_weights=seg_encoder_weight,
            classes=classes_num
        )
        self.loss_fn = smp.losses.SoftCrossEntropyLoss()
        self.metrics = torchmetrics.IoU(num_classes=CLASS_NUM)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=SEG_LR)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        image, mask = batch
        output = self(image)
        loss = self.loss_fn(mask, output)
        self.log('train_loss_step', loss)
        self.log('train_IoU_step', self.metrics(output, mask))
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        losses = torch.cat([loss for loss in outputs['loss']])
        self.log('train_loss_epoch', torch.mean(losses))
        self.log('train_IoU_epoch', self.metrics.compute())

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        output = self(image)
        loss = self.loss_fn(mask, output)
        self.metrics(output, mask)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        losses = torch.cat([loss for loss in outputs['val_loss']])
        self.log('val_loss_epoch', torch.mean(losses))
        self.log('val_IoU_epoch', self.metrics.compute())

    def get_mask(self, x):
        return self(x)
