import torch
from torch import optim

import segmentation_models_pytorch as smp
import pytorch_lightning as pl

import torchmetrics

from config import *


class SegmentationNetwork(pl.LightningModule):
    def __init__(self,
                 seg_encoder=SEG_ENCODER,
                 seg_encoder_depth=SEG_ENCODER_DEPTH,
                 seg_encoder_weight=SEG_ENCODER_WEIGHT,
                 classes=CLASSES
                 ):
        super(SegmentationNetwork, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=seg_encoder,
            encoder_depth=seg_encoder_depth,
            encoder_weights=seg_encoder_weight,
            classes=len(classes),
            activation='softmax'
        )
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass',
                                           classes=len(classes))
        self.metrics = torchmetrics.IoU(num_classes=len(classes))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=SEG_LR)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        image, mask = batch
        output = self(image)
        loss = self.loss_fn(output, mask.long())
        self.log('train_loss_step', loss, on_step=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True)
        self.log('train_IoU_step', self.metrics(preds=output, target=mask))
        return {'loss': loss}

    def training__epoch_end(self, outputs):
        self.log('train_IoU_epoch', self.metrics.compute())

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        output = self(image)
        loss = self.loss_fn(output, mask.long())
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.metrics(preds=output, target=mask)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        self.log('val_IoU', self.metrics.compute())

    def get_mask(self, x):
        return self(x)
