from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

import pytorch_lightning as pl

from transform import triplet_transform

from config import *


class TripletsDataset(Dataset):
    def __init__(self,
                 directory,
                 transform=triplet_transform):
        self.dataset = ImageFolder(directory)
        self.transform = transform
        self.labels = np.array(self.dataset.targets)
        self.images = self.dataset
        self.labels_set = set(self.dataset.class_to_idx.values())
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

    def __getitem__(self, index):
        # TODO: 미완성임 일단 segmentation 먼저 끝내보고, 어떻게 할지 고민해보자!
        anc_img, anc_label = self.images[index][0], self.labels[index].item()
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anc_label])
        negative_label = np.random.choice(list(self.labels_set - set([anc_label])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        pos_img = self.images[positive_index][0]
        pos_label = self.labels[positive_index]
        neg_img = self.images[negative_index][0]
        neg_label = self.labels[negative_index]

        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return (anc_img, pos_img, neg_img), (anc_label, pos_label, neg_label)

    def __len__(self):
        return len(self.dataset)


class TripletDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):
        super(TripletDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = TripletsDataset(directory=f'{DATA_DIR}/train/clothes/')
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        dataset = TripletsDataset(directory=f'{DATA_DIR}/validation/clothes/')
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True)

