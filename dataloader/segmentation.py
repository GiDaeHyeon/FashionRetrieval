import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transform import segmentation_transform
from ..config import *


class SegmentationDataset(Dataset):
    def __init__(self,
                 directory,
                 transform_mode='train'):
        self.transform = segmentation_transform(mode=transform_mode)
        self.images = sorted(os.listdir(directory + '/image'))
        self.masks = sorted(os.listdir(directory + '/mask2'))

        assert len(self.images) == len(self.masks),'데이터셋 점검이 필요합니다.(inconsistency between images and masks)'

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        mask = Image.open(self.masks[index])

        if self.transform is not None:
            augmented = self.transform(image=image,
                                       mask=mask)

        return augmented['image'], augmented['mask']

    def __len__(self):
        return len(self.images)


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=BATCH_SIZE):
        super(SegmentationDataModule, self).__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = SegmentationDataset(directory=f'{DATA_DIR}/train',
                                      transform_mode='train')
        return DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        dataset = SegmentationDataset(directory=f'{DATA_DIR}/validation',
                                      transform_mode='val')
        return DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True)

