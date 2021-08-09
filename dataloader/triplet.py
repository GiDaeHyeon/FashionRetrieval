from torch.utils.data import Dataset
import numpy as np

from transform import triplet_transform


class TripletsDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=triplet_transform):
        self.dataset = dataset
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
            anc_img = self.transform(image=anc_img)['image']
            pos_img = self.transform(image=pos_img)['image']
            neg_img = self.transform(image=neg_img)['image']

        return (anc_img, pos_img, neg_img), (anc_label, pos_label, neg_label)

    def __len__(self):
        return len(self.dataset)