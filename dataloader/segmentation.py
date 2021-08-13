import torch
from glob import glob
import cv2
import numpy as np

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from config import *
from torchvision.transforms import Normalize, ToTensor


class SegmentationDataset(Dataset):
    def __init__(self,
                 directory):
        self.Normalize = Normalize(mean=[.485, .456, .406],
                                   std=[.229, .224, .225])
        self.ToTensor = ToTensor()
        self.images = sorted(glob(directory + '/image/*.jpg'))
        self.masks = sorted(glob(directory + '/mask2/*.png'))

        assert len(self.images) == len(self.masks), \
            f'데이터셋 점검이 필요합니다.(inconsistency between images({len(self.images)}) ' \
            f'and masks({len(self.masks)}))'

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)

        assert self.images[index].split('/')[-1].split('.')[0] == self.masks[index].split('/')[-1].split('.')[0], '데이터셋에 이상이 있습니다.'

        image, mask = cv2.resize(image, dsize=[512, 512], interpolation=cv2.INTER_NEAREST), \
        cv2.resize(mask,  dsize=[512, 512], interpolation=cv2.INTER_NEAREST)

        image, mask = self.ToTensor(image), torch.Tensor(mask).type(torch.int)
        image = self.Normalize(image)

        return image, mask

    def __len__(self):
        return len(self.images)


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):
        super(SegmentationDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = SegmentationDataset(directory=f'{DATA_DIR}/train')
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        dataset = SegmentationDataset(directory=f'{DATA_DIR}/validation')
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True)